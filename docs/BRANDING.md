# Vastloom brand

- **Name:** Vastloom
- **Slogan:** **Run giant models. Keep your machine yours.**

The name combines scale (`vast`) with an adaptive system that weaves model tensors across
memory tiers (`loom`). The visual language uses braided data ribbons, dark graphite hardware,
electric cyan for active movement and restrained violet for depth. It should feel like serious
systems software—not a generic chatbot, spaceship or cryptocurrency product.

## Assets

| Asset | Purpose | Dimensions |
|---|---|---:|
| [`assets/vastloom-hero.png`](assets/vastloom-hero.png) | README and project cover | 1983 × 793 |
| [`assets/vastloom-social-preview.png`](assets/vastloom-social-preview.png) | High-resolution social source | 1774 × 887 |
| [`assets/vastloom-social-preview-1280x640.jpg`](assets/vastloom-social-preview-1280x640.jpg) | GitHub-ready social preview | 1280 × 640 |

The social images are exactly 2:1. Upload the 226,508-byte JPG from **Settings → General →
Social preview**; it follows [GitHub's 1280×640 recommendation and sub-1 MB limit](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).
Keep meaningful alt text when embedding either asset. The high-resolution PNG remains the
editable publication source. These files are small enough for Git; Git LFS is not required.

## Generation prompt set

The committed images were generated with OpenAI's built-in image generation mode from these
final creative briefs and then inspected at full repository resolution.

### README hero

> Create an ultra-wide premium open-source project cover for a native AI inference runtime named
> “VASTLOOM”. Exact visible text: “VASTLOOM” and “Run giant models. Keep your machine yours.” Show
> luminous woven data ribbons flowing through three clearly labelled tiers—GPU / VRAM, SYSTEM RAM,
> NVME STORAGE—toward a refined consumer desktop computer. Dark graphite background, electric cyan
> and restrained violet light, subtle technical grid, cinematic but credible systems-engineering
> aesthetic. Strong editorial typography, generous negative space, crisp at README width. No robot,
> brain, portal, stars, crypto imagery, extra copy, watermark or fake UI. Wide 5:2 composition.

### Social preview

> Create a bold 2:1 GitHub social preview for “VASTLOOM”. Exact visible text: “VASTLOOM” and
> “Run giant models. Keep your machine yours.” Visualize one elegant braided stream weaving through
> three hardware memory layers labelled VRAM, RAM and NVMe, converging into a protected consumer PC.
> Modern dark graphite field, high-contrast cyan and violet signal light, premium open-source systems
> software identity, minimal composition, excellent readability at thumbnail size. No extra words,
> robots, brains, spaceships, portals, crypto motifs, watermark or fake application interface.

## Compatibility naming

New code, commands, packages and generated artifacts use `Vastloom` / `vastloom`. Releases before
`0.4.0-alpha.4` used the former project name. Their immutable persisted identifiers remain accepted
read-only so existing verified model stores and downloads do not need to be rewritten.

Native binary magic `SGTMV001` and heartbeat wire magic `SGHB1` are intentionally stable format
identifiers, not public branding.
