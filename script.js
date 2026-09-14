const repository = "Jesse-Phitidis/BirdWatcher";
const releasePage = `https://github.com/${repository}/releases/latest`;
const releaseStatus = document.querySelector("#release-status");
const downloadCards = document.querySelectorAll(".download-card");

function setFallback(message) {
  releaseStatus.textContent = message;
  downloadCards.forEach((card) => {
    card.href = releasePage;
    card.classList.add("is-fallback");
  });
}

function findAsset(release, platform, extension) {
  const prefix = `BirdWatcher-${platform}-`;
  const suffix = `.${extension}`;
  return release.assets.find((asset) => asset.name.startsWith(prefix) && asset.name.endsWith(suffix));
}

async function loadLatestRelease() {
  try {
    const response = await fetch(`https://api.github.com/repos/${repository}/releases/latest`, {
      headers: { Accept: "application/vnd.github+json" }
    });
    if (!response.ok) throw new Error("Release request failed");

    const release = await response.json();
    let matched = 0;
    downloadCards.forEach((card) => {
      const asset = findAsset(release, card.dataset.platform, card.dataset.extension);
      if (asset) {
        card.href = asset.browser_download_url;
        matched += 1;
      }
    });

    releaseStatus.textContent = matched === downloadCards.length
      ? `Version ${release.tag_name.replace(/^v/, "")}`
      : "Some downloads are available on the release page";
  } catch (error) {
    setFallback("Open the latest release to download");
  }
}

loadLatestRelease();