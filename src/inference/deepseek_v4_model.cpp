#include "vastloom/inference/deepseek_v4_model.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

#include "vastloom/model/imported_checkpoint.hpp"
#include "vastloom/model/safetensors.hpp"
#include "vastloom/util/json.hpp"
#include "vastloom/util/string.hpp"

namespace vastloom::inference {
namespace {

constexpr std::string_view kModelId = "deepseek-ai/DeepSeek-V4-Flash";
constexpr std::string_view kRevision = "60d8d70770c6776ff598c94bb586a859a38244f1";
constexpr ByteCount kTensorPayloadBytes = 159'609'485'896ULL;
constexpr ByteCount kCheckpointFileBytes = 159'617'149'040ULL;
constexpr ByteCount kHeaderSnapshotBytes = 7'663'144ULL;
constexpr ByteCount kRoutedExpertBytes = 150'592'290'816ULL;
constexpr ByteCount kAlwaysActiveBytes = 9'017'195'080ULL;
constexpr ByteCount kBytesPerExpert = 13'369'344ULL;
constexpr std::size_t kTensorCount = 69'187U;
constexpr std::size_t kRoutedExpertTensorCount = 67'584U;
constexpr std::size_t kExpertBundleCount = 11'264U;
constexpr std::size_t kShardCount = 46U;

struct ExpectedFile {
    std::string_view name;
    ByteCount bytes;
    std::string_view sha256;
};

struct ExpectedShard {
    std::string_view name;
    ByteCount bytes;
    std::string_view sha256;
    ByteCount header_snapshot_bytes;
    std::string_view header_sha256;
};

constexpr std::array<ExpectedFile, 5> kMetadataFiles{{
    {"config.json", 1'749U, "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"},
    {"generation_config.json", 170U, "5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0"},
    {"model.safetensors.index.json", 5'371'381U, "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"},
    {"tokenizer.json", 6'367'146U, "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"},
    {"tokenizer_config.json", 801U, "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"},
}};

constexpr std::array<ExpectedShard, kShardCount> kShards{{
    {"model-00001-of-00046.safetensors", 1'059'061'856ULL, "5176586613905d4beaadbaea1cfecd1693e17f9da4a0ea09d99aab7d3f2f5b7c", 96U, "ec94f070d8173f2b3eaae1638b811b8cfa1082a59de9d377081d3008c4cc6247"},
    {"model-00002-of-00046.safetensors", 3'566'321'192ULL, "f04048189d3b472b26d7d02331edd0412e85b184725041c978fdce793a286a0d", 172'240U, "c0460c63141a938c627da59cc153dc79e0840439f7312e421c5026f8a9bca629"},
    {"model-00003-of-00046.safetensors", 3'566'321'192ULL, "df5f80b9b4ca54edf7bda4e640a1b0d1bb95d5995843198dfcce2af0bfc4cb35", 172'240U, "d6b3a0ebca6aa59b76662c1dcc7bbc58fc8d4cdf2ff3fcfeabeb3eb61b1a9c5b"},
    {"model-00004-of-00046.safetensors", 3'596'229'272ULL, "948250b46a6f92df92ef093ab2d0023c31f924232846a84affe84fa3c7794f5c", 173'632U, "a8b814ff108217c6ef71250b2ac9c6e184a5a02abaa10ffcc918a9317bac029a"},
    {"model-00005-of-00046.safetensors", 3'568'768'976ULL, "9fda158bc636215aea4f6834821c81f59eea3733223c874ab66b9f3d6740c4c1", 172'664U, "5ba5fd61cedb5e934a35d4b214d43316b94101ac08e068501179497af4234a7c"},
    {"model-00006-of-00046.safetensors", 3'590'024'776ULL, "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a", 173'552U, "6a227e0a48fb4a9bdc8ad8dd1842d587f949902972bad48ca8ff59f4a6584cc3"},
    {"model-00007-of-00046.safetensors", 3'568'768'976ULL, "2d782c46d6d293189e01ed13ea108de355fd32a9902cea77b264dfc9c4f10c42", 172'664U, "3766bdc1fe3a069f6a7d87ea76e99a2daebb73b616f2cf8d3be8d6cd7a0a61d5"},
    {"model-00008-of-00046.safetensors", 3'590'024'776ULL, "b7d9d8d8932e12ea113f2e83ff412fc1d6b663acedf7f34bbe2bfff41a71c595", 173'552U, "fa087491626731378708d743dcf31dd043c1ba124ac42e04f5d5b6637280c9e7"},
    {"model-00009-of-00046.safetensors", 3'568'768'976ULL, "3197a42d282a8368d4286d11aed52d7fee7928d7950fee4dc18d5154b8111060", 172'664U, "a0d1f5894bcdd9539839ea2bed44941a3062be6a7fec118dcd13fe91fb1c625c"},
    {"model-00010-of-00046.safetensors", 3'590'024'776ULL, "c9cef4200444326f8d802bc1bb5be7c1d15af4bfdec0426db7e841a7bd516a28", 173'552U, "952616c610e5b6f1e88721389f7bc6fd012d88d14e7d1939f9b83a8afd553e2f"},
    {"model-00011-of-00046.safetensors", 3'568'768'976ULL, "916b0b34b713c51dc7a83935ccd9e07c4af017ddfcc5feaeec8faa26f1263f89", 172'664U, "f748fcb4e2d050c65fd87d15ad81122252fdb263aec93312645a668a974d21a5"},
    {"model-00012-of-00046.safetensors", 3'590'026'352ULL, "05739c7d91a302f41a4627587982016a6cc874f875a3ea299d1f2e1dcea5cbb6", 175'128U, "26823212fdefc2d9cde829de94c5069cfaaabee27d65c8b1e810e3f3628153c0"},
    {"model-00013-of-00046.safetensors", 3'568'770'544ULL, "47c5e416b60b9bef9e9005cdad9c991a306ab2dd25a95e1994dda30bd4011905", 174'232U, "17cf3e150276bb1af5983af6d1a3cec23d3bcb6d4d79138c95499fd7ad8a4d58"},
    {"model-00014-of-00046.safetensors", 3'590'026'352ULL, "c881e2671ab45428d824197f174d64fb4403b9530451cb1b0b3899881ab34b7e", 175'128U, "24194492d25306914b45c370fec78253273fb4d77c6cdff862f47a448b60d3f9"},
    {"model-00015-of-00046.safetensors", 3'568'770'544ULL, "cb3daae8e465c5b49f2b439c045971e96cd658fda152882605e9717573d4c8f5", 174'232U, "c5698c00efbc6c0b025dd5c4d3f262c3726f3182ae636511c116c71f71a6a1cb"},
    {"model-00016-of-00046.safetensors", 3'590'026'352ULL, "9eba661fba3162a8a051b0283ad3c11c7c17b33d99684e7029ddf732494bd069", 175'128U, "bfa91569acb70c14d8db4c11f501f012c6b4e41ae286b4aa149f02503ae30fd4"},
    {"model-00017-of-00046.safetensors", 3'568'770'544ULL, "0c36cbc026c5067164cb506ac6522078478feb539fef264289ef70f1d1688f54", 174'232U, "05dc7169809d4a402fb05ddea1fef1ec99659a9e7fad20fb380223ba220e9ac0"},
    {"model-00018-of-00046.safetensors", 3'590'026'352ULL, "1298a07452a409ddbcc5172a4f57a011193bea4f16a6a964c3dd8531cbc06b49", 175'128U, "f68b45f74c5e10a3be1c1a8d83ea874202d136c53a872e884058aa4687f788ae"},
    {"model-00019-of-00046.safetensors", 3'568'770'544ULL, "d3687748aff78adf42b2bbf7469b8df92e1e3e5dcd90c1c050c8fac62d1fe77a", 174'232U, "95055e75864bf5286945a82894ad2640812a61d7583d672f682d1600b723a864"},
    {"model-00020-of-00046.safetensors", 3'590'026'352ULL, "906c652f3c36b510689c2637ebe5172865cc6ccc17515b9fc70ee9e048e7c5af", 175'128U, "72b5179877a51baa5aca4f1db6f8948bc6f5b90b05c7114dd6ff82918bb9413e"},
    {"model-00021-of-00046.safetensors", 3'568'770'544ULL, "f270bf4d0f0067165020baf3c11264a177182918c1ebeec21d2bf33166b44592", 174'232U, "755eb6c67ef5fa846af05c6a1aa03e1ba6b8cb98fda4e9f246cac1c7c2d719c6"},
    {"model-00022-of-00046.safetensors", 3'590'026'352ULL, "d02261b8f1c8d697bac4b23ec9a8423096fb1a1edf923ed1c2faea6cebdb05e8", 175'128U, "f3fa7eeb2c0cc758f8a21933ddb03332acf387888c5cc704ac73679486ba39c8"},
    {"model-00023-of-00046.safetensors", 3'568'770'544ULL, "69fab8bfa1cdfd819382cdef5923dcf10d81879aceb0dc0950be45a6d45771f2", 174'232U, "f7c04ec1b635428abb166d4d316c94f9048cd665f25e0ea5671c0a83f7fb01c9"},
    {"model-00024-of-00046.safetensors", 3'590'026'352ULL, "baba23c06a7b80e108334eb9fe30349de851e822e9b72df756614ae6b5088dbf", 175'128U, "502c67fa0497c5dda867a6ace028d79af986a3212d111e2115dbcdea7f8aacd1"},
    {"model-00025-of-00046.safetensors", 3'568'770'544ULL, "085b7736ebe3d69574930187f8b213f84b35b36852f0a505067f8755e8b11c89", 174'232U, "7d6fdf6cd35b7be3fff2887483e5833f78bd308c7fa23106ed86092564a44e87"},
    {"model-00026-of-00046.safetensors", 3'590'026'352ULL, "fdde6791ab713c93cb256d05d9560d5ecfd2b913563658b62d32af9a17a91546", 175'128U, "575b3670a555390acc3cc96feccfb63f279422f54593949b2ca36adc6c272f4f"},
    {"model-00027-of-00046.safetensors", 3'568'770'544ULL, "2f207b9aef9c56e38f73234e9c5c118c1898676c8ab83d2be0dd410fb196bd01", 174'232U, "885b1bced21faa48df9fa82000681224e8a1d02874e576d788f78a576fe3e2d4"},
    {"model-00028-of-00046.safetensors", 3'590'026'352ULL, "2cc519b5a03a30d45717ffb8408a4f833f3a94f70e35a1de38e95a0ffcdc152e", 175'128U, "34c541ad9720071cadbc14b128c633bf094d6a6616d7885855810d516f4c9240"},
    {"model-00029-of-00046.safetensors", 3'568'770'544ULL, "d10bf34c789f9294d2cc50b695d259dc1d0d5b2303105329be370eb55f0fd882", 174'232U, "428c6a2601872d5461da38dd69b0e5cd2240af6bd826ed1eeace59037e8c4533"},
    {"model-00030-of-00046.safetensors", 3'590'026'352ULL, "0f1c471fa9d9d3612c94039e9efb2a60f367782918f236e95c45ab3fe36c166c", 175'128U, "f7b9ae55931d4d21434164646676f8b8c71053a94f04aca7bd2718ec68eebad5"},
    {"model-00031-of-00046.safetensors", 3'568'770'544ULL, "beafa59d64fae3a5c636be4b2a08956d8d5abaf70856a4a5adb70a200011e43e", 174'232U, "c9d429249b12313ae4e6ca8c54013afd801cc8950d602d32d275d0c5272c71bc"},
    {"model-00032-of-00046.safetensors", 3'590'026'352ULL, "5c6b2934d87ada60493e201652d72075c56ce608091a3d395f3d8e31b6ce036a", 175'128U, "eb00bf03e1e05ddcf32dc655cf3e3f2ae8808684d8062c0e098cadf95e7f2601"},
    {"model-00033-of-00046.safetensors", 3'568'770'544ULL, "c05f917e873d2da0513385c3c30758d29b4b9d244ffde2a5258ae2e6745f23da", 174'232U, "0c6f363d5e14afae9eafbffe5d288b51a21f8041a7d17b07d4a93e69ffe032f8"},
    {"model-00034-of-00046.safetensors", 3'590'026'352ULL, "666b77201ec6946fb1f299f2ebcb462bce53575a97a91fb30db806bbcd099100", 175'128U, "24c9ae647d6a946cc3213becfcbfbdbbb6cb271c5034f42853fe52b47bd12fb4"},
    {"model-00035-of-00046.safetensors", 3'568'770'544ULL, "5e6b9a54fd149ea4908f81cdffd6b09701d9c98e5e7041bb5bd00c6d5c85623f", 174'232U, "774cd0b4dcc8820be658a70e4c9d39563f98ae43029a07b905108df84073cb89"},
    {"model-00036-of-00046.safetensors", 3'590'026'352ULL, "ab72ad9d171fc0867350948e5091878b3c2445a5cfb8a83dd8c25d4272628107", 175'128U, "5a1a058b40405c443fbcb4a3cffab4f817297f3b1bcc4529b8f01ebcc5e2cbf9"},
    {"model-00037-of-00046.safetensors", 3'568'770'544ULL, "93d68bcfc36fdf239f901653c0e96c5d45d8fce4f5be633bbbf93cc75067ec5d", 174'232U, "ab45b66cd8da367dc779e85e9320e2e4eb40a139cfc3cf97d910aeab0c56fd57"},
    {"model-00038-of-00046.safetensors", 3'590'026'352ULL, "809fb799edcf1d9b4511dea68a1ea35c5fc03858e0611a0010913f1c84e66efe", 175'128U, "4e63803bfa0cf285702615fc4ccf6f0b434e6d14bfd416b02887397ebced6cbe"},
    {"model-00039-of-00046.safetensors", 3'568'770'544ULL, "49dba248917454c0a8ac90cc5a012fee625fea6672ae55c40d32a5922a29dd91", 174'232U, "102c53b2de458bb744feb81ba04cde8b712ddea80f10dd6967506ec9b2ce8cbb"},
    {"model-00040-of-00046.safetensors", 3'590'026'352ULL, "09a7b8b6957ff3426d7461dd49a158f4576c0277ab18b78a881094aef29ba84b", 175'128U, "b0d457ee97e5cb77a908b9bb12f1da576b1c9207ca04c443363ae179df5ea90c"},
    {"model-00041-of-00046.safetensors", 3'568'770'544ULL, "a564ac6c6cc7514beadb2e1d9d1fc2baeed5e680ba1b75cd9afc7a50fb90ee85", 174'232U, "668a3efc56bcf1ee261bf151a608eaf7dd637fac4828431e3558dbcace97f12a"},
    {"model-00042-of-00046.safetensors", 3'590'026'352ULL, "bd3f5b898b041559a534c81a2f9bb53a8f9744ee8db37ed11b3a5c9fcf848882", 175'128U, "241b4fab5163b3886ff7517ef03686be804160cd444e1e0085a5a4321a4edfe2"},
    {"model-00043-of-00046.safetensors", 3'568'770'544ULL, "85a414c7991c1276e8db780f6a4390ac25b8f2c7fdb3551831993adb9db69430", 174'232U, "cb19acae159b658e26c3d7bdd166ab861a9cb55bea4320f5c3cff678ee04088b"},
    {"model-00044-of-00046.safetensors", 3'590'026'352ULL, "438b052b8a2d650939e63704f55f1352b946152ba6633cb256c3864ef21d2f62", 175'128U, "e5fb424355c49d76e16d0b18f911793d6dd611babf62821d65a31d2b6340e6c1"},
    {"model-00045-of-00046.safetensors", 1'059'332'516ULL, "9a0fd242134e9ebe4e6993a7631692944838e4fdf20067b3219caa48eab68045", 400U, "ed86fda4ac2c09c307a6578b21cbd610a24da5e6015f75309186c0e9941447cb"},
    {"model-00046-of-00046.safetensors", 3'593'956'092ULL, "f58f722893a6148216a2155cee4a57fe691cea4d3b323135c433a936b932055d", 168'336U, "94ed7b0c3bda457c05f5456aa3a14ac25086f59c82e31fd12c500e93e8e21bcd"},
}};

Result<void> expect_u32(const util::JsonValue& root, std::string_view key, std::uint32_t expected) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "DeepSeek config is missing integer field", std::string(key));
    auto parsed = value->as_u64();
    if (!parsed || parsed.value() != expected) {
        return make_error(ErrorCode::unsupported, "DeepSeek config integer differs from the pinned revision", std::string(key));
    }
    return {};
}

Result<void> expect_number(const util::JsonValue& root, std::string_view key, double expected) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "DeepSeek config is missing numeric field", std::string(key));
    auto parsed = value->as_double();
    if (!parsed || !std::isfinite(parsed.value()) || std::abs(parsed.value() - expected) > 1.0e-12) {
        return make_error(ErrorCode::unsupported, "DeepSeek config number differs from the pinned revision", std::string(key));
    }
    return {};
}

Result<void> expect_string(const util::JsonValue& root, std::string_view key, std::string_view expected) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "DeepSeek config is missing string field", std::string(key));
    auto parsed = value->as_string();
    if (!parsed || parsed.value() != expected) {
        return make_error(ErrorCode::unsupported, "DeepSeek config string differs from the pinned revision", std::string(key));
    }
    return {};
}

Result<void> expect_bool(const util::JsonValue& root, std::string_view key, bool expected) {
    const auto* value = root.find(key);
    if (value == nullptr) return make_error(ErrorCode::invalid_format, "DeepSeek config is missing boolean field", std::string(key));
    auto parsed = value->as_bool();
    if (!parsed || parsed.value() != expected) {
        return make_error(ErrorCode::unsupported, "DeepSeek config boolean differs from the pinned revision", std::string(key));
    }
    return {};
}

Result<void> validate_imported_files(const model::ImportedCheckpoint& imported) {
    constexpr std::size_t expected_count = kMetadataFiles.size() + kShards.size();
    if (imported.files.size() != expected_count) {
        return make_error(ErrorCode::integrity_error, "DeepSeek imported file set is incomplete or contains extras", imported.root.string());
    }
    const auto validate = [&](std::string_view name, ByteCount bytes, std::string_view sha256) -> Result<void> {
        const auto iterator = std::find_if(imported.files.begin(), imported.files.end(), [&](const model::ImportedFile& file) {
            return file.relative_path.generic_string() == name;
        });
        if (iterator == imported.files.end() || iterator->bytes != bytes || iterator->sha256 != sha256) {
            return make_error(ErrorCode::integrity_error, "DeepSeek imported file identity does not match the pinned revision", std::string(name));
        }
        return {};
    };
    for (const auto& file : kMetadataFiles) {
        auto valid = validate(file.name, file.bytes, file.sha256);
        if (!valid) return valid.error();
    }
    for (const auto& shard : kShards) {
        const std::string name = imported.metadata_only
                                     ? "headers/" + std::string(shard.name) + ".header"
                                     : std::string(shard.name);
        auto valid = imported.metadata_only
                         ? validate(name, shard.header_snapshot_bytes, shard.header_sha256)
                         : validate(name, shard.bytes, shard.sha256);
        if (!valid) return valid.error();
    }
    return {};
}

std::string shard_name(std::uint32_t index) {
    std::string number = std::to_string(index);
    return "model-" + std::string(5U - number.size(), '0') + number + "-of-00046.safetensors";
}

using ExpectedTensorMap = std::map<std::string, std::string, std::less<>>;

ExpectedTensorMap expected_tensor_map(const DeepseekV4Config& config) {
    ExpectedTensorMap names;
    const auto add = [&](std::string name, std::string shard) { names.emplace(std::move(name), std::move(shard)); };
    add("embed.weight", shard_name(1U));
    for (const auto name : {"hc_head_base", "hc_head_fn", "hc_head_scale", "norm.weight", "head.weight"}) {
        add(name, shard_name(45U));
    }

    constexpr std::array<std::string_view, 28> common{{
        "hc_attn_base", "hc_ffn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_scale",
        "attn.attn_sink", "attn.wq_a.weight", "attn.wq_a.scale", "attn.wq_b.weight", "attn.wq_b.scale",
        "attn.q_norm.weight", "attn.wo_a.weight", "attn.wo_a.scale", "attn.wkv.weight", "attn.wkv.scale",
        "attn.kv_norm.weight", "attn.wo_b.weight", "attn.wo_b.scale", "attn_norm.weight", "ffn_norm.weight",
        "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale", "ffn.shared_experts.w3.weight",
        "ffn.shared_experts.w3.scale", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w2.scale",
        "ffn.gate.weight",
    }};
    constexpr std::array<std::string_view, 6> expert{{
        "w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale",
    }};
    constexpr std::array<std::string_view, 4> compressor{{
        "attn.compressor.ape", "attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
        "attn.compressor.norm.weight",
    }};
    constexpr std::array<std::string_view, 7> indexer{{
        "attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale", "attn.indexer.compressor.ape",
        "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.norm.weight", "attn.indexer.weights_proj.weight",
    }};
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        const std::string shard = shard_name(layer + 2U);
        for (const auto suffix : common) add(prefix + std::string(suffix), shard);
        for (std::uint32_t expert_id = 0; expert_id < config.num_routed_experts; ++expert_id) {
            const std::string expert_prefix = prefix + "ffn.experts." + std::to_string(expert_id) + ".";
            for (const auto suffix : expert) add(expert_prefix + std::string(suffix), shard);
        }
        if (layer < config.num_hash_layers) add(prefix + "ffn.gate.tid2eid", shard);
        else add(prefix + "ffn.gate.bias", shard);
        if (layer >= 2U) {
            for (const auto suffix : compressor) add(prefix + std::string(suffix), shard);
            if ((layer % 2U) == 0U) {
                for (const auto suffix : indexer) add(prefix + std::string(suffix), shard);
            }
        }
    }

    const std::string mtp_prefix = "mtp.0.";
    const std::string mtp_shard = shard_name(46U);
    constexpr std::array<std::string_view, 39> mtp_common{{
        "hc_head_base", "hc_head_fn", "hc_head_scale", "hc_attn_base", "hc_ffn_base", "hc_attn_fn",
        "hc_attn_scale", "hc_ffn_fn", "hc_ffn_scale", "attn.attn_sink", "attn.wq_a.weight", "attn.wq_a.scale",
        "attn.wq_b.weight", "attn.wq_b.scale", "attn.q_norm.weight", "attn.wo_a.weight", "attn.wo_a.scale",
        "attn.wkv.weight", "attn.wkv.scale", "attn.kv_norm.weight", "attn.wo_b.weight", "attn.wo_b.scale",
        "attn_norm.weight", "ffn_norm.weight", "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w3.weight", "ffn.shared_experts.w3.scale", "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w2.scale", "ffn.gate.bias", "ffn.gate.weight", "e_proj.weight", "e_proj.scale",
        "h_proj.weight", "h_proj.scale", "enorm.weight", "hnorm.weight", "norm.weight",
    }};
    for (const auto suffix : mtp_common) add(mtp_prefix + std::string(suffix), mtp_shard);
    for (std::uint32_t expert_id = 0; expert_id < config.num_routed_experts; ++expert_id) {
        const std::string expert_prefix = mtp_prefix + "ffn.experts." + std::to_string(expert_id) + ".";
        for (const auto suffix : expert) add(expert_prefix + std::string(suffix), mtp_shard);
    }
    return names;
}

Result<ExpectedTensorMap> load_and_validate_index(const std::filesystem::path& path,
                                                  const DeepseekV4Config& config) {
    util::JsonLimits limits;
    limits.max_input_bytes = 8U * MiB;
    limits.max_nodes = 70'000U;
    limits.max_object_members = 70'000U;
    limits.max_array_elements = 16U;
    auto json = util::load_json_file(path, limits);
    if (!json) return json.error();
    const auto* metadata = json.value().find("metadata");
    const auto* map_value = json.value().find("weight_map");
    if (metadata == nullptr || map_value == nullptr) {
        return make_error(ErrorCode::invalid_format, "DeepSeek weight index is missing metadata or weight_map", path.string());
    }
    const auto* total_value = metadata->find("total_size");
    auto map = map_value->as_object();
    auto total = total_value == nullptr ? Result<std::uint64_t>(make_error(ErrorCode::invalid_format, "missing total_size"))
                                        : total_value->as_u64();
    if (!map || !total || total.value() != kTensorPayloadBytes) {
        return make_error(ErrorCode::integrity_error, "DeepSeek weight index payload identity differs from the pinned revision", path.string());
    }
    auto expected = expected_tensor_map(config);
    if (expected.size() != kTensorCount || map.value()->size() != expected.size()) {
        return make_error(ErrorCode::integrity_error, "DeepSeek tensor count differs from the pinned architecture", path.string());
    }
    for (const auto& [name, shard] : expected) {
        const auto iterator = map.value()->find(name);
        if (iterator == map.value()->end()) {
            return make_error(ErrorCode::integrity_error, "DeepSeek weight index is missing an expected tensor", name);
        }
        auto actual_shard = iterator->second.as_string();
        if (!actual_shard || actual_shard.value() != shard) {
            return make_error(ErrorCode::integrity_error, "DeepSeek tensor maps to an unexpected shard", name);
        }
    }
    return expected;
}

bool routed_expert_tensor(std::string_view name) {
    return (name.starts_with("layers.") || name.starts_with("mtp.0.")) &&
           name.find(".ffn.experts.") != std::string_view::npos;
}

Result<std::uint32_t> parse_identifier(std::string_view text,
                                       std::size_t begin,
                                       std::size_t end,
                                       std::string_view tensor_name) {
    if (begin >= end || end > text.size()) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek tensor has an invalid numeric identity",
                          std::string(tensor_name));
    }
    std::uint32_t value = 0U;
    const auto parsed = std::from_chars(text.data() + begin, text.data() + end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + end) {
        return make_error(ErrorCode::invalid_format,
                          "DeepSeek tensor has an invalid numeric identity",
                          std::string(tensor_name));
    }
    return value;
}

struct TensorIdentity {
    model::TensorRole role{model::TensorRole::other};
    std::optional<std::uint32_t> layer;
    std::optional<std::uint32_t> expert;
    bool speculative{false};
};

Result<TensorIdentity> classify_tensor(std::string_view name, const DeepseekV4Config& config) {
    TensorIdentity identity;
    if (name.starts_with("layers.")) {
        constexpr std::size_t begin = std::string_view("layers.").size();
        const auto end = name.find('.', begin);
        if (end == std::string_view::npos) {
            return make_error(ErrorCode::invalid_format, "DeepSeek layer tensor name is malformed", std::string(name));
        }
        auto layer = parse_identifier(name, begin, end, name);
        if (!layer || layer.value() >= config.num_hidden_layers) {
            return make_error(ErrorCode::invalid_format, "DeepSeek layer tensor identity is out of range", std::string(name));
        }
        identity.layer = layer.value();
    } else if (name.starts_with("mtp.")) {
        constexpr std::size_t begin = std::string_view("mtp.").size();
        const auto end = name.find('.', begin);
        if (end == std::string_view::npos) {
            return make_error(ErrorCode::invalid_format, "DeepSeek MTP tensor name is malformed", std::string(name));
        }
        auto mtp_layer = parse_identifier(name, begin, end, name);
        if (!mtp_layer || mtp_layer.value() >= config.num_nextn_predict_layers ||
            mtp_layer.value() > std::numeric_limits<std::uint32_t>::max() - config.num_hidden_layers) {
            return make_error(ErrorCode::invalid_format, "DeepSeek MTP tensor identity is out of range", std::string(name));
        }
        identity.layer = config.num_hidden_layers + mtp_layer.value();
        identity.speculative = true;
    }

    constexpr std::string_view expert_marker = ".ffn.experts.";
    const auto expert_marker_at = name.find(expert_marker);
    if (expert_marker_at != std::string_view::npos) {
        const auto begin = expert_marker_at + expert_marker.size();
        const auto end = name.find('.', begin);
        if (!identity.layer.has_value() || end == std::string_view::npos) {
            return make_error(ErrorCode::invalid_format, "DeepSeek routed expert tensor name is malformed", std::string(name));
        }
        auto expert = parse_identifier(name, begin, end, name);
        if (!expert || expert.value() >= config.num_routed_experts) {
            return make_error(ErrorCode::invalid_format, "DeepSeek routed expert identity is out of range", std::string(name));
        }
        identity.expert = expert.value();
        identity.role = model::TensorRole::routed_expert;
    } else if (name.find(".ffn.shared_experts.") != std::string_view::npos) {
        identity.role = model::TensorRole::shared_expert;
    } else if (name.find(".ffn.gate.") != std::string_view::npos) {
        identity.role = model::TensorRole::router;
    } else if (name == "embed.weight") {
        identity.role = model::TensorRole::embedding;
    } else if (name == "head.weight") {
        identity.role = model::TensorRole::output_head;
    } else if (name.find(".attn.") != std::string_view::npos) {
        identity.role = model::TensorRole::attention;
    } else if (name.starts_with("hc_") || name.find(".hc_") != std::string_view::npos) {
        identity.role = model::TensorRole::hyper_connection;
    } else if (name.find("norm.") != std::string_view::npos) {
        identity.role = model::TensorRole::normalization;
    } else if (identity.speculative) {
        identity.role = model::TensorRole::speculative;
    }
    return identity;
}

std::string expert_identity(std::string_view name) {
    const auto marker = name.find(".ffn.experts.");
    if (marker == std::string_view::npos) return {};
    const auto id_begin = marker + std::string_view(".ffn.experts.").size();
    const auto id_end = name.find('.', id_begin);
    if (id_end == std::string_view::npos) return {};
    return std::string(name.substr(0U, id_end));
}

struct CatalogSummary {
    std::vector<DeepseekDTypeSummary> dtypes;
    ByteCount total_tensor_bytes{0};
    ByteCount routed_expert_bytes{0};
    std::size_t routed_expert_tensors{0};
    model::LogicalTensorCatalog logical_catalog;
};

Result<CatalogSummary> validate_headers(const std::filesystem::path& root,
                                        bool metadata_only,
                                        const ExpectedTensorMap& expected,
                                        const DeepseekV4Config& config) {
    std::map<model::TensorDType, DeepseekDTypeSummary> dtypes;
    std::set<std::string, std::less<>> seen;
    std::map<std::string, ByteCount, std::less<>> expert_bytes;
    CatalogSummary summary;
    summary.logical_catalog.model_id = std::string(kModelId);
    summary.logical_catalog.revision = std::string(kRevision);
    summary.logical_catalog.metadata_only = metadata_only;
    summary.logical_catalog.tensors.reserve(expected.size());
    for (const auto& shard : kShards) {
        const auto path = metadata_only
                              ? root / "headers" / (std::string(shard.name) + ".header")
                              : root / std::string(shard.name);
        auto opened = metadata_only
                          ? model::SafetensorsFile::open_header_snapshot(path, shard.bytes, 1U * MiB, 2'000U)
                          : model::SafetensorsFile::open(path, 1U * MiB, 2'000U);
        if (!opened) return opened.error();
        if (opened.value().file_bytes() != shard.bytes ||
            opened.value().header_bytes() + 8U != shard.header_snapshot_bytes) {
            return make_error(ErrorCode::integrity_error, "DeepSeek Safetensors header identity differs from the pinned shard", path.string());
        }
        for (const auto& tensor : opened.value().tensors()) {
            const auto expected_tensor = expected.find(tensor.name);
            if (expected_tensor == expected.end() || expected_tensor->second != shard.name) {
                return make_error(ErrorCode::integrity_error, "DeepSeek header contains an unexpected tensor or shard mapping", tensor.name);
            }
            if (!seen.insert(tensor.name).second) {
                return make_error(ErrorCode::integrity_error, "DeepSeek tensor is duplicated across headers", tensor.name);
            }
            if (tensor.byte_count > std::numeric_limits<ByteCount>::max() - summary.total_tensor_bytes) {
                return make_error(ErrorCode::invalid_format, "DeepSeek tensor byte count overflows", tensor.name);
            }
            summary.total_tensor_bytes += tensor.byte_count;
            auto& dtype = dtypes[tensor.dtype];
            dtype.dtype = std::string(model::to_string(tensor.dtype));
            ++dtype.tensor_count;
            dtype.bytes += tensor.byte_count;
            if (routed_expert_tensor(tensor.name)) {
                ++summary.routed_expert_tensors;
                summary.routed_expert_bytes += tensor.byte_count;
                expert_bytes[expert_identity(tensor.name)] += tensor.byte_count;
            }
            auto identity = classify_tensor(tensor.name, config);
            if (!identity) return identity.error();
            summary.logical_catalog.tensors.push_back(model::LogicalTensorDescriptor{
                tensor.name,
                tensor.dtype,
                tensor.shape,
                identity.value().role,
                identity.value().layer,
                identity.value().expert,
                identity.value().speculative,
                model::TensorSourceRange{root / std::string(shard.name), tensor.absolute_offset, tensor.byte_count},
                std::nullopt,
            });
        }
    }
    if (seen.size() != expected.size() || summary.total_tensor_bytes != kTensorPayloadBytes ||
        summary.routed_expert_tensors != kRoutedExpertTensorCount ||
        summary.routed_expert_bytes != kRoutedExpertBytes || expert_bytes.size() != 11'264U) {
        return make_error(ErrorCode::integrity_error, "DeepSeek Safetensors header catalog totals differ from the pinned revision", root.string());
    }
    for (const auto& [name, bytes] : expert_bytes) {
        if (name.empty() || bytes != kBytesPerExpert) {
            return make_error(ErrorCode::integrity_error, "DeepSeek routed expert byte layout differs from the pinned revision", name);
        }
    }
    auto logical = model::finalize_logical_tensor_catalog(summary.logical_catalog, kTensorCount, kExpertBundleCount);
    if (!logical) return logical.error();
    if (summary.logical_catalog.tensor_bytes != kTensorPayloadBytes ||
        summary.logical_catalog.routed_expert_bytes != kRoutedExpertBytes ||
        summary.logical_catalog.always_active_bytes != kAlwaysActiveBytes ||
        summary.logical_catalog.experts.size() != kExpertBundleCount) {
        return make_error(ErrorCode::integrity_error,
                          "DeepSeek logical tensor catalog differs from the pinned revision",
                          root.string());
    }
    for (const auto& expert : summary.logical_catalog.experts) {
        if (expert.bytes != kBytesPerExpert || expert.tensor_indices.size() != 6U ||
            expert.tensor_slices.size() != 6U || expert.read_ranges.empty()) {
            return make_error(ErrorCode::integrity_error,
                              "DeepSeek logical expert bundle differs from the pinned revision",
                              root.string());
        }
    }
    constexpr std::array<std::pair<model::TensorDType, std::pair<std::size_t, ByteCount>>, 6> expected_dtypes{{
        {model::TensorDType::bf16, {433U, 2'830'518'528ULL}},
        {model::TensorDType::f32, {417U, 144'672'072ULL}},
        {model::TensorDType::f8_e4m3, {375U, 6'023'020'544ULL}},
        {model::TensorDType::f8_e8m0, {34'167U, 8'858'737'664ULL}},
        {model::TensorDType::i8, {33'792U, 141'733'920'768ULL}},
        {model::TensorDType::i64, {3U, 18'616'320ULL}},
    }};
    if (dtypes.size() != expected_dtypes.size()) {
        return make_error(ErrorCode::integrity_error, "DeepSeek Safetensors dtype set differs from the pinned revision", root.string());
    }
    for (const auto& [type, identity] : expected_dtypes) {
        const auto iterator = dtypes.find(type);
        if (iterator == dtypes.end() || iterator->second.tensor_count != identity.first ||
            iterator->second.bytes != identity.second) {
            return make_error(ErrorCode::integrity_error, "DeepSeek Safetensors dtype totals differ from the pinned revision", std::string(model::to_string(type)));
        }
        summary.dtypes.push_back(iterator->second);
    }
    return summary;
}

}  // namespace

Result<DeepseekV4Config> load_deepseek_v4_config(const std::filesystem::path& config_path) {
    util::JsonLimits limits;
    limits.max_input_bytes = 1U * MiB;
    limits.max_nodes = 4'096U;
    auto json = util::load_json_file(config_path, limits);
    if (!json) return json.error();
    if (!json.value().is_object()) {
        return make_error(ErrorCode::invalid_format, "DeepSeek config root is not an object", config_path.string());
    }
    const auto* architectures_value = json.value().find("architectures");
    auto architectures = architectures_value == nullptr
                             ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing architectures"))
                             : architectures_value->as_array();
    if (!architectures || architectures.value()->size() != 1U || !(*architectures.value())[0].as_string() ||
        (*architectures.value())[0].as_string().value() != "DeepseekV4ForCausalLM") {
        return make_error(ErrorCode::unsupported, "Checkpoint is not the pinned DeepSeek V4 Flash architecture", config_path.string());
    }

    const std::array<Result<void>, 22> checks{{
        expect_string(json.value(), "model_type", "deepseek_v4"), expect_string(json.value(), "hidden_act", "silu"),
        expect_string(json.value(), "torch_dtype", "bfloat16"), expect_string(json.value(), "expert_dtype", "fp4"),
        expect_string(json.value(), "topk_method", "noaux_tc"), expect_string(json.value(), "scoring_func", "sqrtsoftplus"),
        expect_bool(json.value(), "attention_bias", false), expect_bool(json.value(), "tie_word_embeddings", false),
        expect_bool(json.value(), "norm_topk_prob", true), expect_bool(json.value(), "use_cache", true),
        expect_u32(json.value(), "vocab_size", 129'280U), expect_u32(json.value(), "hidden_size", 4'096U),
        expect_u32(json.value(), "moe_intermediate_size", 2'048U), expect_u32(json.value(), "num_hidden_layers", 43U),
        expect_u32(json.value(), "num_attention_heads", 64U), expect_u32(json.value(), "num_key_value_heads", 1U),
        expect_u32(json.value(), "head_dim", 512U), expect_u32(json.value(), "max_position_embeddings", 1'048'576U),
        expect_u32(json.value(), "n_routed_experts", 256U), expect_u32(json.value(), "n_shared_experts", 1U),
        expect_u32(json.value(), "num_experts_per_tok", 6U), expect_u32(json.value(), "num_hash_layers", 3U),
    }};
    for (const auto& check : checks) if (!check) return check.error();
    const std::array<Result<void>, 22> remaining{{
        expect_u32(json.value(), "num_nextn_predict_layers", 1U), expect_u32(json.value(), "q_lora_rank", 1'024U),
        expect_u32(json.value(), "o_lora_rank", 1'024U), expect_u32(json.value(), "qk_rope_head_dim", 64U),
        expect_u32(json.value(), "index_head_dim", 128U), expect_u32(json.value(), "index_n_heads", 64U),
        expect_u32(json.value(), "index_topk", 512U), expect_u32(json.value(), "sliding_window", 128U),
        expect_u32(json.value(), "bos_token_id", 0U), expect_u32(json.value(), "eos_token_id", 1U),
        expect_number(json.value(), "rms_norm_eps", 1.0e-6), expect_number(json.value(), "rope_theta", 10'000.0),
        expect_number(json.value(), "routed_scaling_factor", 1.5), expect_number(json.value(), "swiglu_limit", 10.0),
        expect_u32(json.value(), "compress_rope_theta", 160'000U), expect_u32(json.value(), "hc_mult", 4U),
        expect_number(json.value(), "attention_dropout", 0.0), expect_number(json.value(), "hc_eps", 1.0e-6),
        expect_u32(json.value(), "hc_sinkhorn_iters", 20U), expect_number(json.value(), "initializer_range", 0.02),
        expect_u32(json.value(), "o_groups", 8U), expect_string(json.value(), "transformers_version", "4.57.1"),
    }};
    for (const auto& check : remaining) if (!check) return check.error();

    const auto* ratios_value = json.value().find("compress_ratios");
    auto ratios = ratios_value == nullptr
                      ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing compress_ratios"))
                      : ratios_value->as_array();
    if (!ratios || ratios.value()->size() != 44U) {
        return make_error(ErrorCode::unsupported, "DeepSeek compression schedule differs from the pinned revision", config_path.string());
    }
    for (std::size_t index = 0; index < ratios.value()->size(); ++index) {
        const std::uint64_t expected = index < 2U || index == 43U ? 0U : ((index % 2U) == 0U ? 4U : 128U);
        auto ratio = (*ratios.value())[index].as_u64();
        if (!ratio || ratio.value() != expected) {
            return make_error(ErrorCode::unsupported, "DeepSeek compression schedule differs from the pinned revision", std::to_string(index));
        }
    }

    const auto* quantization = json.value().find("quantization_config");
    const auto* rope_scaling = json.value().find("rope_scaling");
    if (quantization == nullptr || !quantization->is_object() || rope_scaling == nullptr || !rope_scaling->is_object()) {
        return make_error(ErrorCode::invalid_format, "DeepSeek quantization or RoPE metadata is missing", config_path.string());
    }
    const std::array<Result<void>, 9> nested{{
        expect_string(*quantization, "quant_method", "fp8"), expect_string(*quantization, "fmt", "e4m3"),
        expect_string(*quantization, "scale_fmt", "ue8m0"), expect_string(*quantization, "activation_scheme", "dynamic"),
        expect_string(*rope_scaling, "type", "yarn"), expect_u32(*rope_scaling, "factor", 16U),
        expect_u32(*rope_scaling, "original_max_position_embeddings", 65'536U), expect_u32(*rope_scaling, "beta_fast", 32U),
        expect_u32(*rope_scaling, "beta_slow", 1U),
    }};
    for (const auto& check : nested) if (!check) return check.error();
    const auto* block_value = quantization->find("weight_block_size");
    auto block = block_value == nullptr
                     ? Result<const util::JsonValue::Array*>(make_error(ErrorCode::invalid_format, "missing weight_block_size"))
                     : block_value->as_array();
    if (!block || block.value()->size() != 2U || !(*block.value())[0].as_u64() || !(*block.value())[1].as_u64() ||
        (*block.value())[0].as_u64().value() != 128U || (*block.value())[1].as_u64().value() != 128U) {
        return make_error(ErrorCode::unsupported, "DeepSeek quantization block size differs from the pinned revision", config_path.string());
    }

    return DeepseekV4Config{
        129'280U, 4'096U, 2'048U, 43U, 64U, 1U, 512U, 1'048'576U, 256U, 1U, 6U, 3U, 1U,
        1'024U, 1'024U, 128U, 64U, 512U, 128U, 1U, 0U, 1.0e-6F, 10'000.0F, 1.5F,
    };
}

Result<DeepseekV4Inspection> inspect_deepseek_v4_checkpoint(const std::filesystem::path& model_directory,
                                                             bool verify_hashes,
                                                             ByteCount safe_ram_cap_bytes) {
    auto imported = model::verify_imported_checkpoint(model_directory, verify_hashes);
    if (!imported) return imported.error();
    if (imported.value().model_id != kModelId || imported.value().revision != kRevision) {
        return make_error(ErrorCode::unsupported, "Imported checkpoint is not the pinned DeepSeek V4 Flash revision", model_directory.string());
    }
    auto files = validate_imported_files(imported.value());
    if (!files) return files.error();
    auto config = load_deepseek_v4_config(imported.value().root / "config.json");
    if (!config) return config.error();
    auto index = load_and_validate_index(imported.value().root / "model.safetensors.index.json", config.value());
    if (!index) return index.error();
    auto headers = validate_headers(imported.value().root, imported.value().metadata_only, index.value(), config.value());
    if (!headers) return headers.error();

    DeepseekV4Inspection result;
    result.root = imported.value().root;
    result.model_id = imported.value().model_id;
    result.revision = imported.value().revision;
    result.config = config.value();
    result.metadata_only = imported.value().metadata_only;
    result.hashes_verified = imported.value().full_hash_verified;
    result.header_catalog_complete = true;
    result.logical_catalog_valid = true;
    result.execution_supported = false;
    result.shard_count = kShardCount;
    result.tensor_count = kTensorCount;
    result.routed_expert_tensor_count = headers.value().routed_expert_tensors;
    result.imported_file_bytes = imported.value().total_bytes;
    result.checkpoint_file_bytes = kCheckpointFileBytes;
    result.header_snapshot_bytes = kHeaderSnapshotBytes;
    result.tensor_payload_bytes = headers.value().total_tensor_bytes;
    result.routed_expert_bytes = headers.value().routed_expert_bytes;
    result.always_active_bytes = kAlwaysActiveBytes;
    result.bytes_per_expert = kBytesPerExpert;
    result.active_expert_bytes_per_layer = kBytesPerExpert * config.value().num_experts_per_token;
    result.no_cache_expert_bytes_per_token = result.active_expert_bytes_per_layer * config.value().num_hidden_layers;
    result.safe_ram_cap_bytes = safe_ram_cap_bytes;
    result.dtypes = std::move(headers.value().dtypes);
    result.logical_catalog = std::move(headers.value().logical_catalog);
    return result;
}

std::string to_json(const DeepseekV4Inspection& inspection) {
    std::ostringstream output;
    output << "{\"ok\":true,\"model_id\":\"" << util::json_escape(inspection.model_id)
           << "\",\"revision\":\"" << util::json_escape(inspection.revision)
           << "\",\"architecture\":\"DeepseekV4ForCausalLM\",\"metadata_only\":"
           << (inspection.metadata_only ? "true" : "false")
           << ",\"hashes_verified\":" << (inspection.hashes_verified ? "true" : "false")
           << ",\"header_catalog_complete\":" << (inspection.header_catalog_complete ? "true" : "false")
           << ",\"logical_catalog_valid\":" << (inspection.logical_catalog_valid ? "true" : "false")
           << ",\"execution_supported\":" << (inspection.execution_supported ? "true" : "false")
           << ",\"layers\":" << inspection.config.num_hidden_layers
           << ",\"hidden_size\":" << inspection.config.hidden_size
           << ",\"attention_heads\":" << inspection.config.num_attention_heads
           << ",\"kv_heads\":" << inspection.config.num_key_value_heads
           << ",\"context\":" << inspection.config.max_position_embeddings
           << ",\"routed_experts\":" << inspection.config.num_routed_experts
           << ",\"shared_experts\":" << inspection.config.num_shared_experts
           << ",\"experts_per_token\":" << inspection.config.num_experts_per_token
           << ",\"hash_layers\":" << inspection.config.num_hash_layers
           << ",\"mtp_layers\":" << inspection.config.num_nextn_predict_layers
           << ",\"shard_count\":" << inspection.shard_count
           << ",\"tensor_count\":" << inspection.tensor_count
           << ",\"routed_expert_tensor_count\":" << inspection.routed_expert_tensor_count
           << ",\"expert_bundle_count\":" << inspection.logical_catalog.experts.size()
           << ",\"expert_read_range_count\":" << inspection.logical_catalog.expert_read_range_count
           << ",\"imported_file_bytes\":" << inspection.imported_file_bytes
           << ",\"checkpoint_file_bytes\":" << inspection.checkpoint_file_bytes
           << ",\"header_snapshot_bytes\":" << inspection.header_snapshot_bytes
           << ",\"tensor_payload_bytes\":" << inspection.tensor_payload_bytes
           << ",\"routed_expert_bytes\":" << inspection.routed_expert_bytes
           << ",\"always_active_bytes\":" << inspection.always_active_bytes
           << ",\"bytes_per_expert\":" << inspection.bytes_per_expert
           << ",\"active_expert_bytes_per_layer\":" << inspection.active_expert_bytes_per_layer
           << ",\"no_cache_expert_bytes_per_token\":" << inspection.no_cache_expert_bytes_per_token
           << ",\"safe_ram_cap_bytes\":" << inspection.safe_ram_cap_bytes
           << ",\"dtypes\":[";
    for (std::size_t index = 0; index < inspection.dtypes.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& dtype = inspection.dtypes[index];
        output << "{\"dtype\":\"" << util::json_escape(dtype.dtype) << "\",\"tensor_count\":"
               << dtype.tensor_count << ",\"bytes\":" << dtype.bytes << '}';
    }
    output << "],\"warning\":\"Inspection support only; DeepSeek execution is not implemented\"}";
    return output.str();
}

std::string to_text(const DeepseekV4Inspection& inspection) {
    std::ostringstream output;
    output << "Model: " << inspection.model_id << '\n'
           << "Revision: " << inspection.revision << '\n'
           << "Architecture: 43 layers, width 4096, 64 heads / 1 KV head, 256 routed experts, top-6\n"
           << "Checkpoint state: " << (inspection.metadata_only ? "verified metadata plus header snapshots" : "complete checkpoint") << '\n'
           << "Safetensors: " << inspection.shard_count << " shards, " << inspection.tensor_count << " tensors, "
           << format_bytes(inspection.tensor_payload_bytes) << " payload\n"
           << "Header snapshots: " << format_bytes(inspection.header_snapshot_bytes) << '\n'
           << "Logical expert bundles: " << inspection.logical_catalog.experts.size() << " in "
           << inspection.logical_catalog.expert_read_range_count << " bounded source ranges\n"
           << "Always-active tensors: " << format_bytes(inspection.always_active_bytes) << '\n'
           << "Routed experts: " << format_bytes(inspection.routed_expert_bytes) << '\n'
           << "One routed expert: " << format_bytes(inspection.bytes_per_expert) << '\n'
           << "Active routed experts per layer: " << format_bytes(inspection.active_expert_bytes_per_layer) << '\n'
           << "No-cache routed bytes per main-model token: " << format_bytes(inspection.no_cache_expert_bytes_per_token) << '\n'
           << "Dtypes:";
    for (const auto& dtype : inspection.dtypes) {
        output << " " << dtype.dtype << '=' << dtype.tensor_count << "/" << format_bytes(dtype.bytes);
    }
    output << "\nExecution: NOT IMPLEMENTED (inspection support only)\n";
    return output.str();
}

}  // namespace vastloom::inference
