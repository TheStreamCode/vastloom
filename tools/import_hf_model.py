#!/usr/bin/env python3
"""Pinned, bounded and atomic Hugging Face checkpoint importer for Vastloom."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Iterable
from urllib.parse import quote
import http.client
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if os.name == "nt":
    import msvcrt
else:
    import fcntl


GIB = 1024**3
DEFAULT_DISK_RESERVE = 300 * GIB
LARGE_DOWNLOAD_THRESHOLD = 20 * GIB
COPY_BUFFER_BYTES = 1024 * 1024
STORE_MARKER_NAME = ".vastloom-store"
LEGACY_STORE_MARKER_NAME = ".stargate-store"
MARKER = "format=vastloom.store.v1\noriginal_weights=immutable\ncache=reconstructable\n"
LEGACY_MARKER = "format=stargate.store.v1\noriginal_weights=immutable\ncache=reconstructable\n"
IMPORT_MANIFEST_NAME = "vastloom-import.json"
LEGACY_IMPORT_MANIFEST_NAME = "stargate-import.json"
IMPORT_FORMAT = "vastloom.hf-import.v1"
LEGACY_IMPORT_FORMAT = "stargate.hf-import.v1"


def progress(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    sha256: str | None = None
    weight: bool = False
    source_name: str | None = None
    metadata_only: bool = False
    range_prefix: bool = False
    source_size: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    revision: str
    files: tuple[FileSpec, ...]
    artifact_name: str | None = None


SMOLLM2 = ModelSpec(
    alias="smollm2",
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    files=(
        FileSpec("config.json", 861),
        FileSpec("generation_config.json", 132),
        FileSpec("merges.txt", 466_391),
        FileSpec(
            "model.safetensors",
            269_060_552,
            "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
            True,
        ),
        FileSpec("special_tokens_map.json", 655),
        FileSpec("tokenizer.json", 2_104_556),
        FileSpec("tokenizer_config.json", 3_764),
        FileSpec("vocab.json", 800_662),
    ),
)

OLMOE = ModelSpec(
    alias="olmoe",
    model_id="allenai/OLMoE-1B-7B-0924-Instruct",
    revision="7f1c97f440f06ce36705e4f2b843edb5925f4498",
    files=(
        FileSpec("config.json", 759, "42ea6f5fe31e9243290ba3c348a4ad615b88a6fbc6e10544e21a87b6dff5da6c"),
        FileSpec("generation_config.json", 120, "77cf3f0caaa5e6d6972d910c70ca03f64fc47eccb4374082bdc8fc7b2c524a45"),
        FileSpec(
            "model-00001-of-00003.safetensors",
            4_997_744_872,
            "6c79ac0487c3f23e8ee3d38752197d1e4a1a39d6c1438ec5fd7862874bb19321",
            True,
        ),
        FileSpec(
            "model-00002-of-00003.safetensors",
            4_997_235_176,
            "b58ca7a9f28f35d76b56b0ec40fc46c356fd84ccf078083b251a3ad6a2da9a35",
            True,
        ),
        FileSpec(
            "model-00003-of-00003.safetensors",
            3_843_741_912,
            "e93653ed509223e63eabae6f72ac8ea5a41115c5c7785574b9b2068ca0961c45",
            True,
        ),
        FileSpec(
            "model.safetensors.index.json",
            287_214,
            "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
        ),
        FileSpec(
            "special_tokens_map.json",
            433,
            "cf22141a130514760f6280166b8d1c9a56610578188aad0790eb6eaf6ea1eb08",
        ),
        FileSpec(
            "tokenizer.json",
            2_115_598,
            "b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167",
        ),
        FileSpec(
            "tokenizer_config.json",
            5_986,
            "d3079f9c26a02fd28ee8d9bdfea4e96da7f5e803e06707ec8d1511acaaac7864",
        ),
    ),
)


DEEPSEEK_V4_FLASH_WEIGHTS = (
    FileSpec("model-00001-of-00046.safetensors", 1_059_061_856, "5176586613905d4beaadbaea1cfecd1693e17f9da4a0ea09d99aab7d3f2f5b7c", True),
    FileSpec("model-00002-of-00046.safetensors", 3_566_321_192, "f04048189d3b472b26d7d02331edd0412e85b184725041c978fdce793a286a0d", True),
    FileSpec("model-00003-of-00046.safetensors", 3_566_321_192, "df5f80b9b4ca54edf7bda4e640a1b0d1bb95d5995843198dfcce2af0bfc4cb35", True),
    FileSpec("model-00004-of-00046.safetensors", 3_596_229_272, "948250b46a6f92df92ef093ab2d0023c31f924232846a84affe84fa3c7794f5c", True),
    FileSpec("model-00005-of-00046.safetensors", 3_568_768_976, "9fda158bc636215aea4f6834821c81f59eea3733223c874ab66b9f3d6740c4c1", True),
    FileSpec("model-00006-of-00046.safetensors", 3_590_024_776, "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a", True),
    FileSpec("model-00007-of-00046.safetensors", 3_568_768_976, "2d782c46d6d293189e01ed13ea108de355fd32a9902cea77b264dfc9c4f10c42", True),
    FileSpec("model-00008-of-00046.safetensors", 3_590_024_776, "b7d9d8d8932e12ea113f2e83ff412fc1d6b663acedf7f34bbe2bfff41a71c595", True),
    FileSpec("model-00009-of-00046.safetensors", 3_568_768_976, "3197a42d282a8368d4286d11aed52d7fee7928d7950fee4dc18d5154b8111060", True),
    FileSpec("model-00010-of-00046.safetensors", 3_590_024_776, "c9cef4200444326f8d802bc1bb5be7c1d15af4bfdec0426db7e841a7bd516a28", True),
    FileSpec("model-00011-of-00046.safetensors", 3_568_768_976, "916b0b34b713c51dc7a83935ccd9e07c4af017ddfcc5feaeec8faa26f1263f89", True),
    FileSpec("model-00012-of-00046.safetensors", 3_590_026_352, "05739c7d91a302f41a4627587982016a6cc874f875a3ea299d1f2e1dcea5cbb6", True),
    FileSpec("model-00013-of-00046.safetensors", 3_568_770_544, "47c5e416b60b9bef9e9005cdad9c991a306ab2dd25a95e1994dda30bd4011905", True),
    FileSpec("model-00014-of-00046.safetensors", 3_590_026_352, "c881e2671ab45428d824197f174d64fb4403b9530451cb1b0b3899881ab34b7e", True),
    FileSpec("model-00015-of-00046.safetensors", 3_568_770_544, "cb3daae8e465c5b49f2b439c045971e96cd658fda152882605e9717573d4c8f5", True),
    FileSpec("model-00016-of-00046.safetensors", 3_590_026_352, "9eba661fba3162a8a051b0283ad3c11c7c17b33d99684e7029ddf732494bd069", True),
    FileSpec("model-00017-of-00046.safetensors", 3_568_770_544, "0c36cbc026c5067164cb506ac6522078478feb539fef264289ef70f1d1688f54", True),
    FileSpec("model-00018-of-00046.safetensors", 3_590_026_352, "1298a07452a409ddbcc5172a4f57a011193bea4f16a6a964c3dd8531cbc06b49", True),
    FileSpec("model-00019-of-00046.safetensors", 3_568_770_544, "d3687748aff78adf42b2bbf7469b8df92e1e3e5dcd90c1c050c8fac62d1fe77a", True),
    FileSpec("model-00020-of-00046.safetensors", 3_590_026_352, "906c652f3c36b510689c2637ebe5172865cc6ccc17515b9fc70ee9e048e7c5af", True),
    FileSpec("model-00021-of-00046.safetensors", 3_568_770_544, "f270bf4d0f0067165020baf3c11264a177182918c1ebeec21d2bf33166b44592", True),
    FileSpec("model-00022-of-00046.safetensors", 3_590_026_352, "d02261b8f1c8d697bac4b23ec9a8423096fb1a1edf923ed1c2faea6cebdb05e8", True),
    FileSpec("model-00023-of-00046.safetensors", 3_568_770_544, "69fab8bfa1cdfd819382cdef5923dcf10d81879aceb0dc0950be45a6d45771f2", True),
    FileSpec("model-00024-of-00046.safetensors", 3_590_026_352, "baba23c06a7b80e108334eb9fe30349de851e822e9b72df756614ae6b5088dbf", True),
    FileSpec("model-00025-of-00046.safetensors", 3_568_770_544, "085b7736ebe3d69574930187f8b213f84b35b36852f0a505067f8755e8b11c89", True),
    FileSpec("model-00026-of-00046.safetensors", 3_590_026_352, "fdde6791ab713c93cb256d05d9560d5ecfd2b913563658b62d32af9a17a91546", True),
    FileSpec("model-00027-of-00046.safetensors", 3_568_770_544, "2f207b9aef9c56e38f73234e9c5c118c1898676c8ab83d2be0dd410fb196bd01", True),
    FileSpec("model-00028-of-00046.safetensors", 3_590_026_352, "2cc519b5a03a30d45717ffb8408a4f833f3a94f70e35a1de38e95a0ffcdc152e", True),
    FileSpec("model-00029-of-00046.safetensors", 3_568_770_544, "d10bf34c789f9294d2cc50b695d259dc1d0d5b2303105329be370eb55f0fd882", True),
    FileSpec("model-00030-of-00046.safetensors", 3_590_026_352, "0f1c471fa9d9d3612c94039e9efb2a60f367782918f236e95c45ab3fe36c166c", True),
    FileSpec("model-00031-of-00046.safetensors", 3_568_770_544, "beafa59d64fae3a5c636be4b2a08956d8d5abaf70856a4a5adb70a200011e43e", True),
    FileSpec("model-00032-of-00046.safetensors", 3_590_026_352, "5c6b2934d87ada60493e201652d72075c56ce608091a3d395f3d8e31b6ce036a", True),
    FileSpec("model-00033-of-00046.safetensors", 3_568_770_544, "c05f917e873d2da0513385c3c30758d29b4b9d244ffde2a5258ae2e6745f23da", True),
    FileSpec("model-00034-of-00046.safetensors", 3_590_026_352, "666b77201ec6946fb1f299f2ebcb462bce53575a97a91fb30db806bbcd099100", True),
    FileSpec("model-00035-of-00046.safetensors", 3_568_770_544, "5e6b9a54fd149ea4908f81cdffd6b09701d9c98e5e7041bb5bd00c6d5c85623f", True),
    FileSpec("model-00036-of-00046.safetensors", 3_590_026_352, "ab72ad9d171fc0867350948e5091878b3c2445a5cfb8a83dd8c25d4272628107", True),
    FileSpec("model-00037-of-00046.safetensors", 3_568_770_544, "93d68bcfc36fdf239f901653c0e96c5d45d8fce4f5be633bbbf93cc75067ec5d", True),
    FileSpec("model-00038-of-00046.safetensors", 3_590_026_352, "809fb799edcf1d9b4511dea68a1ea35c5fc03858e0611a0010913f1c84e66efe", True),
    FileSpec("model-00039-of-00046.safetensors", 3_568_770_544, "49dba248917454c0a8ac90cc5a012fee625fea6672ae55c40d32a5922a29dd91", True),
    FileSpec("model-00040-of-00046.safetensors", 3_590_026_352, "09a7b8b6957ff3426d7461dd49a158f4576c0277ab18b78a881094aef29ba84b", True),
    FileSpec("model-00041-of-00046.safetensors", 3_568_770_544, "a564ac6c6cc7514beadb2e1d9d1fc2baeed5e680ba1b75cd9afc7a50fb90ee85", True),
    FileSpec("model-00042-of-00046.safetensors", 3_590_026_352, "bd3f5b898b041559a534c81a2f9bb53a8f9744ee8db37ed11b3a5c9fcf848882", True),
    FileSpec("model-00043-of-00046.safetensors", 3_568_770_544, "85a414c7991c1276e8db780f6a4390ac25b8f2c7fdb3551831993adb9db69430", True),
    FileSpec("model-00044-of-00046.safetensors", 3_590_026_352, "438b052b8a2d650939e63704f55f1352b946152ba6633cb256c3864ef21d2f62", True),
    FileSpec("model-00045-of-00046.safetensors", 1_059_332_516, "9a0fd242134e9ebe4e6993a7631692944838e4fdf20067b3219caa48eab68045", True),
    FileSpec("model-00046-of-00046.safetensors", 3_593_956_092, "f58f722893a6148216a2155cee4a57fe691cea4d3b323135c433a936b932055d", True),
)

DEEPSEEK_V4_FLASH_HEADER_IDENTITIES = (
    ("model-00001-of-00046.safetensors", 96, "ec94f070d8173f2b3eaae1638b811b8cfa1082a59de9d377081d3008c4cc6247"),
    ("model-00002-of-00046.safetensors", 172_240, "c0460c63141a938c627da59cc153dc79e0840439f7312e421c5026f8a9bca629"),
    ("model-00003-of-00046.safetensors", 172_240, "d6b3a0ebca6aa59b76662c1dcc7bbc58fc8d4cdf2ff3fcfeabeb3eb61b1a9c5b"),
    ("model-00004-of-00046.safetensors", 173_632, "a8b814ff108217c6ef71250b2ac9c6e184a5a02abaa10ffcc918a9317bac029a"),
    ("model-00005-of-00046.safetensors", 172_664, "5ba5fd61cedb5e934a35d4b214d43316b94101ac08e068501179497af4234a7c"),
    ("model-00006-of-00046.safetensors", 173_552, "6a227e0a48fb4a9bdc8ad8dd1842d587f949902972bad48ca8ff59f4a6584cc3"),
    ("model-00007-of-00046.safetensors", 172_664, "3766bdc1fe3a069f6a7d87ea76e99a2daebb73b616f2cf8d3be8d6cd7a0a61d5"),
    ("model-00008-of-00046.safetensors", 173_552, "fa087491626731378708d743dcf31dd043c1ba124ac42e04f5d5b6637280c9e7"),
    ("model-00009-of-00046.safetensors", 172_664, "a0d1f5894bcdd9539839ea2bed44941a3062be6a7fec118dcd13fe91fb1c625c"),
    ("model-00010-of-00046.safetensors", 173_552, "952616c610e5b6f1e88721389f7bc6fd012d88d14e7d1939f9b83a8afd553e2f"),
    ("model-00011-of-00046.safetensors", 172_664, "f748fcb4e2d050c65fd87d15ad81122252fdb263aec93312645a668a974d21a5"),
    ("model-00012-of-00046.safetensors", 175_128, "26823212fdefc2d9cde829de94c5069cfaaabee27d65c8b1e810e3f3628153c0"),
    ("model-00013-of-00046.safetensors", 174_232, "17cf3e150276bb1af5983af6d1a3cec23d3bcb6d4d79138c95499fd7ad8a4d58"),
    ("model-00014-of-00046.safetensors", 175_128, "24194492d25306914b45c370fec78253273fb4d77c6cdff862f47a448b60d3f9"),
    ("model-00015-of-00046.safetensors", 174_232, "c5698c00efbc6c0b025dd5c4d3f262c3726f3182ae636511c116c71f71a6a1cb"),
    ("model-00016-of-00046.safetensors", 175_128, "bfa91569acb70c14d8db4c11f501f012c6b4e41ae286b4aa149f02503ae30fd4"),
    ("model-00017-of-00046.safetensors", 174_232, "05dc7169809d4a402fb05ddea1fef1ec99659a9e7fad20fb380223ba220e9ac0"),
    ("model-00018-of-00046.safetensors", 175_128, "f68b45f74c5e10a3be1c1a8d83ea874202d136c53a872e884058aa4687f788ae"),
    ("model-00019-of-00046.safetensors", 174_232, "95055e75864bf5286945a82894ad2640812a61d7583d672f682d1600b723a864"),
    ("model-00020-of-00046.safetensors", 175_128, "72b5179877a51baa5aca4f1db6f8948bc6f5b90b05c7114dd6ff82918bb9413e"),
    ("model-00021-of-00046.safetensors", 174_232, "755eb6c67ef5fa846af05c6a1aa03e1ba6b8cb98fda4e9f246cac1c7c2d719c6"),
    ("model-00022-of-00046.safetensors", 175_128, "f3fa7eeb2c0cc758f8a21933ddb03332acf387888c5cc704ac73679486ba39c8"),
    ("model-00023-of-00046.safetensors", 174_232, "f7c04ec1b635428abb166d4d316c94f9048cd665f25e0ea5671c0a83f7fb01c9"),
    ("model-00024-of-00046.safetensors", 175_128, "502c67fa0497c5dda867a6ace028d79af986a3212d111e2115dbcdea7f8aacd1"),
    ("model-00025-of-00046.safetensors", 174_232, "7d6fdf6cd35b7be3fff2887483e5833f78bd308c7fa23106ed86092564a44e87"),
    ("model-00026-of-00046.safetensors", 175_128, "575b3670a555390acc3cc96feccfb63f279422f54593949b2ca36adc6c272f4f"),
    ("model-00027-of-00046.safetensors", 174_232, "885b1bced21faa48df9fa82000681224e8a1d02874e576d788f78a576fe3e2d4"),
    ("model-00028-of-00046.safetensors", 175_128, "34c541ad9720071cadbc14b128c633bf094d6a6616d7885855810d516f4c9240"),
    ("model-00029-of-00046.safetensors", 174_232, "428c6a2601872d5461da38dd69b0e5cd2240af6bd826ed1eeace59037e8c4533"),
    ("model-00030-of-00046.safetensors", 175_128, "f7b9ae55931d4d21434164646676f8b8c71053a94f04aca7bd2718ec68eebad5"),
    ("model-00031-of-00046.safetensors", 174_232, "c9d429249b12313ae4e6ca8c54013afd801cc8950d602d32d275d0c5272c71bc"),
    ("model-00032-of-00046.safetensors", 175_128, "eb00bf03e1e05ddcf32dc655cf3e3f2ae8808684d8062c0e098cadf95e7f2601"),
    ("model-00033-of-00046.safetensors", 174_232, "0c6f363d5e14afae9eafbffe5d288b51a21f8041a7d17b07d4a93e69ffe032f8"),
    ("model-00034-of-00046.safetensors", 175_128, "24c9ae647d6a946cc3213becfcbfbdbbb6cb271c5034f42853fe52b47bd12fb4"),
    ("model-00035-of-00046.safetensors", 174_232, "774cd0b4dcc8820be658a70e4c9d39563f98ae43029a07b905108df84073cb89"),
    ("model-00036-of-00046.safetensors", 175_128, "5a1a058b40405c443fbcb4a3cffab4f817297f3b1bcc4529b8f01ebcc5e2cbf9"),
    ("model-00037-of-00046.safetensors", 174_232, "ab45b66cd8da367dc779e85e9320e2e4eb40a139cfc3cf97d910aeab0c56fd57"),
    ("model-00038-of-00046.safetensors", 175_128, "4e63803bfa0cf285702615fc4ccf6f0b434e6d14bfd416b02887397ebced6cbe"),
    ("model-00039-of-00046.safetensors", 174_232, "102c53b2de458bb744feb81ba04cde8b712ddea80f10dd6967506ec9b2ce8cbb"),
    ("model-00040-of-00046.safetensors", 175_128, "b0d457ee97e5cb77a908b9bb12f1da576b1c9207ca04c443363ae179df5ea90c"),
    ("model-00041-of-00046.safetensors", 174_232, "668a3efc56bcf1ee261bf151a608eaf7dd637fac4828431e3558dbcace97f12a"),
    ("model-00042-of-00046.safetensors", 175_128, "241b4fab5163b3886ff7517ef03686be804160cd444e1e0085a5a4321a4edfe2"),
    ("model-00043-of-00046.safetensors", 174_232, "cb19acae159b658e26c3d7bdd166ab861a9cb55bea4320f5c3cff678ee04088b"),
    ("model-00044-of-00046.safetensors", 175_128, "e5fb424355c49d76e16d0b18f911793d6dd611babf62821d65a31d2b6340e6c1"),
    ("model-00045-of-00046.safetensors", 400, "ed86fda4ac2c09c307a6578b21cbd610a24da5e6015f75309186c0e9941447cb"),
    ("model-00046-of-00046.safetensors", 168_336, "94ed7b0c3bda457c05f5456aa3a14ac25086f59c82e31fd12c500e93e8e21bcd"),
)


def _deepseek_header_files() -> tuple[FileSpec, ...]:
    source_sizes = {item.name: item.size for item in DEEPSEEK_V4_FLASH_WEIGHTS}
    return tuple(
        FileSpec(
            name=f"headers/{name}.header",
            size=size,
            sha256=sha256,
            source_name=name,
            metadata_only=True,
            range_prefix=True,
            source_size=source_sizes[name],
        )
        for name, size, sha256 in DEEPSEEK_V4_FLASH_HEADER_IDENTITIES
    )


DEEPSEEK_V4_FLASH = ModelSpec(
    alias="deepseek-v4-flash",
    model_id="deepseek-ai/DeepSeek-V4-Flash",
    revision="60d8d70770c6776ff598c94bb586a859a38244f1",
    files=(
        FileSpec("config.json", 1_749, "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"),
        FileSpec("generation_config.json", 170, "5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0"),
        *DEEPSEEK_V4_FLASH_WEIGHTS,
        FileSpec("model.safetensors.index.json", 5_371_381, "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"),
        FileSpec("tokenizer.json", 6_367_146, "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"),
        FileSpec("tokenizer_config.json", 801, "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"),
        *_deepseek_header_files(),
    ),
)


DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S = ModelSpec(
    alias="deepseek-v4-flash-0731-gguf-iq1s",
    model_id="unsloth/DeepSeek-V4-Flash-0731-GGUF",
    revision="fbbb5b93fb787c21338159b0af3318bb3f4d9768",
    files=(
        # Split 1 contains only GGUF metadata and is small enough to import in full for both
        # modes. Splits 2 and 3 are immutable payload objects in a full import and bounded,
        # independently hashed header prefixes in metadata-only mode.
        FileSpec(
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf",
            5_257_664,
            "2fa152e25a14500e42e6d98f20d200eec1e945f34dc1cfb981b651b9c86b97be",
        ),
        FileSpec(
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf",
            49_093_726_624,
            "e9fc77544cd48aa0c31fbf32fdf5432f2b560b92a9aeffd9f690d99867cf0e5f",
            True,
        ),
        FileSpec(
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf",
            33_440_253_504,
            "fd99d5cd0503d668b6febc2eedcc9db68e264cfb79dd355efc88c15eabe4248d",
            True,
        ),
        FileSpec(
            "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf.header",
            52_320,
            "7128e5c180640517eb10a2ba2eb22a5153f80b3d8316ec70122246a5a2f3db47",
            source_name="UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf",
            metadata_only=True,
            range_prefix=True,
            source_size=49_093_726_624,
        ),
        FileSpec(
            "headers/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf.header",
            33_600,
            "48c9b417ec3c389e834bb299a3273376e8dabbc2855116d12cadb92977091717",
            source_name="UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf",
            metadata_only=True,
            range_prefix=True,
            source_size=33_440_253_504,
        ),
    ),
    # Keep nested cache/staging paths below legacy Windows MAX_PATH without weakening the
    # model/revision identity recorded in the immutable manifest.
    artifact_name="deepseek-v4-flash-0731-iq1s-fbbb5b93",
)


DEEPSEEK_V4_FLASH_0731_GGUF_IQ2XXS = ModelSpec(
    alias="deepseek-v4-flash-0731-gguf-iq2xxs",
    model_id="unsloth/DeepSeek-V4-Flash-0731-GGUF",
    revision="fbbb5b93fb787c21338159b0af3318bb3f4d9768",
    files=(
        # Split 1 contains only GGUF metadata and is small enough to import in full for both
        # modes. Splits 2 and 3 are immutable payload objects in a full import and bounded,
        # independently hashed 2 MiB header prefixes in metadata-only mode. The prefixes carry
        # the complete GGUF metadata/tensor-info block (46,912 and 39,008 bytes respectively).
        FileSpec(
            "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf",
            5_257_664,
            "c58c9d62eac7b62e9578b52613f425e48313d7212ab8d1d76caed8ea8de26595",
        ),
        FileSpec(
            "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf",
            49_890_588_800,
            "65a113df6d4469f16db6882b6919e153c464c3c78c833f5e1b41a33803cdbd52",
            True,
        ),
        FileSpec(
            "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00003-of-00003.gguf",
            40_964_890_464,
            "a69102ddfaf4a84426e11fdb66716654f4260dc3a1de3ade9fd50e006b8691d3",
            True,
        ),
        FileSpec(
            "headers/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf.header",
            2_097_152,
            "dbbfe94ed10ffb3d98e3e96878c6aca4289b9a82c5d13f5fc3e2593eda06ba16",
            source_name="UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf",
            metadata_only=True,
            range_prefix=True,
            source_size=49_890_588_800,
        ),
        FileSpec(
            "headers/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00003-of-00003.gguf.header",
            2_097_152,
            "162724024b4d9aa11b07f22d1555a34a23a657ea96def7a4f0ad92dd81f4cc32",
            source_name="UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00003-of-00003.gguf",
            metadata_only=True,
            range_prefix=True,
            source_size=40_964_890_464,
        ),
    ),
    # Keep nested cache/staging paths below legacy Windows MAX_PATH without weakening the
    # model/revision identity recorded in the immutable manifest.
    artifact_name="deepseek-v4-flash-0731-iq2xxs-fbbb5b93",
)


MODELS = {
    item.alias: item
    for item in (
        SMOLLM2,
        OLMOE,
        DEEPSEEK_V4_FLASH,
        DEEPSEEK_V4_FLASH_0731_GGUF_IQ1S,
        DEEPSEEK_V4_FLASH_0731_GGUF_IQ2XXS,
    )
}


def validate_store(store: Path) -> Path:
    root = store.resolve(strict=True)
    current_marker = root / STORE_MARKER_NAME
    legacy_marker = root / LEGACY_STORE_MARKER_NAME
    markers = [path for path in (current_marker, legacy_marker) if path.exists() or path.is_symlink()]
    if len(markers) != 1:
        raise RuntimeError(f"invalid Vastloom model store: {root}")
    marker = markers[0]
    expected = MARKER if marker == current_marker else LEGACY_MARKER
    if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"invalid Vastloom model store: {root}")
    original = (root / "models" / "original").resolve(strict=True)
    if original.parent.parent != root or original.is_symlink():
        raise RuntimeError(f"unsafe Vastloom original-model directory: {original}")
    cache = (root / "cache").resolve(strict=True)
    if cache.parent != root or cache.is_symlink():
        raise RuntimeError(f"unsafe Vastloom cache directory: {cache}")
    return root


@contextmanager
def import_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(f"another import owns the checkpoint lock: {path}") from error
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def target_name(spec: ModelSpec, metadata_only: bool) -> str:
    model = spec.artifact_name or spec.model_id.replace("/", "-")
    if not model or Path(model).name != model or model in (".", ".."):
        raise RuntimeError("unsafe importer artifact name")
    suffix = "--metadata" if metadata_only else ""
    revision = "" if spec.artifact_name else f"--{spec.revision}"
    return f"{model}{revision}{suffix}"


def selected_files(spec: ModelSpec, metadata_only: bool) -> tuple[FileSpec, ...]:
    if metadata_only:
        return tuple(item for item in spec.files if not item.weight)
    return tuple(item for item in spec.files if not item.metadata_only)


def preflight_disk(root: Path, required_bytes: int, reserve_bytes: int) -> None:
    if required_bytes < 0 or reserve_bytes < 20 * GIB:
        raise RuntimeError("invalid download size or disk reserve")
    available = shutil.disk_usage(root).free
    if available < required_bytes + reserve_bytes:
        raise RuntimeError(
            f"download refused: {available} bytes free, {required_bytes} required, "
            f"and {reserve_bytes} reserved"
        )


def file_url(spec: ModelSpec, item: FileSpec) -> str:
    repository = quote(spec.model_id, safe="/")
    revision = quote(spec.revision, safe="")
    filename = quote(item.source_name or item.name, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def lower_process_priority() -> None:
    """Best-effort priority reduction; safety must not depend on it succeeding."""
    try:
        if os.name == "nt":
            below_normal_priority_class = 0x00004000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.SetPriorityClass.restype = ctypes.c_int
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class):
                raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
        else:
            os.nice(5)
    except OSError:
        pass


def validate_existing(target: Path, spec: ModelSpec, files: Iterable[FileSpec], metadata_only: bool) -> None:
    expected_files = tuple(files)
    current_manifest = target / IMPORT_MANIFEST_NAME
    legacy_manifest = target / LEGACY_IMPORT_MANIFEST_NAME
    manifests = [path for path in (current_manifest, legacy_manifest) if path.exists() or path.is_symlink()]
    if len(manifests) != 1:
        raise RuntimeError(f"existing immutable target has no import manifest: {target}")
    manifest_path = manifests[0]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"existing immutable target has an unsafe import manifest: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") not in (IMPORT_FORMAT, LEGACY_IMPORT_FORMAT)
        or manifest.get("model_id") != spec.model_id
        or manifest.get("revision") != spec.revision
        or bool(manifest.get("metadata_only")) != metadata_only
    ):
        raise RuntimeError(f"existing immutable target identity does not match: {target}")
    recorded = {item["name"]: item for item in manifest.get("files", [])}
    if len(recorded) != len(manifest.get("files", [])) or len(recorded) != len(expected_files):
        raise RuntimeError(f"existing immutable artifact has an unexpected file set: {target}")
    for item in expected_files:
        path = target / item.name
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"existing immutable artifact is missing or has the wrong size: {path}")
        expected_hash = item.sha256 or recorded.get(item.name, {}).get("sha256")
        if not expected_hash or hash_file(path) != expected_hash:
            raise RuntimeError(f"existing immutable artifact failed SHA-256 validation: {path}")


DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_BACKOFF_SECONDS = 5.0


def _is_transient_download_error(error: BaseException) -> bool:
    if isinstance(error, HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, (URLError, TimeoutError, ConnectionError, http.client.HTTPException))


def download_file(
    url: str,
    partial: Path,
    item: FileSpec,
    timeout: float,
    max_bytes_per_second: int,
) -> str:
    # A multi-gigabyte shard must survive a flaky link: a transient transport failure
    # (reset, timeout, truncated response) retries with resume - the partial file is
    # preserved and the next attempt re-opens with a Range header. Clear protocol
    # refusals (4xx) are not retried.
    attempt = 1
    while True:
        try:
            return _download_file_attempt(url, partial, item, timeout, max_bytes_per_second)
        except (HTTPError, URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if attempt >= DOWNLOAD_ATTEMPTS or not _is_transient_download_error(error):
                raise
            delay = DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt
            progress(
                f"transient download error for {item.name} ({error}); "
                f"resuming in {delay:.0f}s (attempt {attempt + 1}/{DOWNLOAD_ATTEMPTS})"
            )
            time.sleep(delay)
            attempt += 1


def _download_file_attempt(
    url: str,
    partial: Path,
    item: FileSpec,
    timeout: float,
    max_bytes_per_second: int,
) -> str:
    digest = hashlib.sha256()
    received = partial.stat().st_size if partial.exists() else 0
    if received > item.size:
        raise RuntimeError(f"partial download exceeds declared size for {item.name}")
    if received:
        progress(f"rehash partial {item.name}: {received}/{item.size} bytes")
        with partial.open("rb") as existing:
            while chunk := existing.read(COPY_BUFFER_BYTES):
                digest.update(chunk)
    initial_bytes = received
    started = time.monotonic()
    next_report = ((received // GIB) + 1) * GIB
    progress(f"download {item.name}: {received}/{item.size} bytes")
    headers = {"User-Agent": "vastloom-local-ai/0.4-importer"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if item.range_prefix:
        if item.source_size is None or item.source_size < item.size:
            raise RuntimeError(f"invalid pinned range source size for {item.name}")
        headers["Range"] = f"bytes={received}-{item.size - 1}"
    elif received:
        headers["Range"] = f"bytes={received}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if (received or item.range_prefix) and status != 206:
            raise RuntimeError(f"server refused safe resume for {item.name}; partial file was preserved")
        if item.range_prefix:
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {received}-{item.size - 1}/{item.source_size}"
            if content_range != expected_range:
                raise RuntimeError(f"server returned an invalid bounded range for {item.name}; partial file was preserved")
        elif received:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {received}-"):
                raise RuntimeError(f"server returned an invalid resume range for {item.name}; partial file was preserved")
        mode = "ab" if received else "xb"
        with partial.open(mode) as output:
            while True:
                chunk = response.read(COPY_BUFFER_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > item.size:
                    raise RuntimeError(f"server sent more bytes than declared for {item.name}")
                output.write(chunk)
                digest.update(chunk)
                if received >= next_report:
                    progress(f"download {item.name}: {received}/{item.size} bytes")
                    next_report += GIB
                if max_bytes_per_second > 0:
                    transferred = received - initial_bytes
                    target_elapsed = transferred / max_bytes_per_second
                    delay = target_elapsed - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
            output.flush()
            os.fsync(output.fileno())
    if received != item.size:
        raise RuntimeError(f"short download for {item.name}: expected {item.size}, got {received}")
    actual_hash = digest.hexdigest()
    if item.sha256 and actual_hash != item.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {item.name}")
    progress(f"verified {item.name}: {received} bytes")
    return actual_hash


def import_model(
    store: Path,
    spec: ModelSpec,
    *,
    metadata_only: bool,
    reserve_bytes: int,
    timeout: float,
    dry_run: bool,
    max_bytes_per_second: int = 128 * 1024 * 1024,
    allow_large_download: bool = False,
) -> Path:
    root = validate_store(store)
    files = selected_files(spec, metadata_only)
    weight_bytes = sum(item.size for item in files if item.weight)
    if weight_bytes >= LARGE_DOWNLOAD_THRESHOLD and not allow_large_download:
        raise RuntimeError(
            f"large checkpoint import refused: {weight_bytes} weight bytes require "
            "--allow-large-download after reviewing storage headroom"
        )
    original = root / "models" / "original"
    target = original / target_name(spec, metadata_only)
    if target.exists():
        validate_existing(target, spec, files, metadata_only)
        return target
    if dry_run:
        preflight_disk(root, sum(item.size for item in files), reserve_bytes)
        return target

    download_root = root / "cache" / "imports" / target.name
    download_root.mkdir(parents=True, exist_ok=True)
    if download_root.resolve(strict=True).parent.parent != (root / "cache").resolve(strict=True):
        raise RuntimeError("unsafe importer download-cache path")
    preserved_bytes = 0
    for item in files:
        cached = download_root / item.name
        partial = cached.with_name(cached.name + ".partial")
        if cached.exists():
            preserved_bytes += min(cached.stat().st_size, item.size)
        elif partial.exists():
            preserved_bytes += min(partial.stat().st_size, item.size)
    preflight_disk(root, sum(item.size for item in files) - preserved_bytes, reserve_bytes)
    lock_path = download_root / ".import.lock"
    incoming = original / f".incoming-{target.name}-{os.getpid()}-{time.time_ns()}"
    if incoming.parent.resolve(strict=True) != original.resolve(strict=True):
        raise RuntimeError("unsafe importer staging path")
    try:
        with import_lock(lock_path):
            records: list[dict[str, object]] = []
            cached_files: list[Path] = []
            for item in files:
                cached = download_root / item.name
                partial = cached.with_name(cached.name + ".partial")
                sidecar = cached.with_name(cached.name + ".sha256")
                cached.parent.mkdir(parents=True, exist_ok=True)
                if cached.exists():
                    expected_hash = item.sha256
                    if expected_hash is None and sidecar.is_file():
                        expected_hash = sidecar.read_text(encoding="ascii").strip()
                    if cached.stat().st_size != item.size or not expected_hash or hash_file(cached) != expected_hash:
                        raise RuntimeError(f"cached artifact failed validation; preserved for inspection: {cached}")
                    actual_hash = expected_hash
                    progress(f"reused verified cache {item.name}: {item.size} bytes")
                else:
                    already_received = partial.stat().st_size if partial.exists() else 0
                    preflight_disk(root, item.size - already_received, reserve_bytes)
                    actual_hash = download_file(
                        file_url(spec, item), partial, item, timeout, max_bytes_per_second
                    )
                    os.replace(partial, cached)
                    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
                    sidecar_partial.write_text(actual_hash + "\n", encoding="ascii", newline="\n")
                    os.replace(sidecar_partial, sidecar)
                records.append({"name": item.name, "size": item.size, "sha256": actual_hash})
                cached_files.append(cached)

            incoming.mkdir(mode=0o700)
            for item, cached in zip(files, cached_files, strict=True):
                destination = incoming / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(cached, destination)

            manifest = {
                "format": IMPORT_FORMAT,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "metadata_only": metadata_only,
                "files": records,
            }
            manifest_partial = incoming / f"{IMPORT_MANIFEST_NAME}.partial"
            with manifest_partial.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(manifest_partial, incoming / IMPORT_MANIFEST_NAME)
            incoming.rename(target)

            for cached in cached_files:
                cached.unlink()
                cached.with_name(cached.name + ".sha256").unlink(missing_ok=True)
            for path in target.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IREAD)
            return target
    except BaseException:
        if incoming.exists() and incoming.parent.resolve(strict=True) == original.resolve(strict=True):
            for path in incoming.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(incoming)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--reserve-gib", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-mib-per-sec", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="explicitly authorize checkpoints with at least 20 GiB of weight payload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reserve_gib < 20 or args.timeout <= 0 or args.max_mib_per_sec < 1 or args.max_mib_per_sec > 1024:
        print("ERROR: reserve, timeout or bandwidth limit is outside safe bounds", file=sys.stderr)
        return 64
    lower_process_priority()
    try:
        target = import_model(
            args.store,
            MODELS[args.model],
            metadata_only=args.metadata_only,
            reserve_bytes=args.reserve_gib * GIB,
            timeout=args.timeout,
            dry_run=args.dry_run,
            max_bytes_per_second=args.max_mib_per_sec * 1024 * 1024,
            allow_large_download=args.allow_large_download,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
