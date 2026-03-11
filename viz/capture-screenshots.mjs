#!/usr/bin/env node
/**
 * Capture PNG screenshots of the interactive superposition visualizations.
 *
 * Usage:
 *   npx playwright test --config=- < /dev/null  # won't work, use direct API
 *   node viz/capture-screenshots.mjs
 *
 * Requires: npx playwright install chromium
 */
import { chromium } from "playwright";
import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.join(__dirname, "output");

const pages = [
  {
    file: "superposition-p5.html",
    out: "superposition-p5-screenshot.png",
    width: 800,
    height: 700,
    waitMs: 3000, // let p5 render a few frames
  },
  {
    file: "superposition-d3.html",
    out: "superposition-d3-screenshot.png",
    width: 820,
    height: 700,
    waitMs: 4000, // let d3 accumulate some density
  },
];

// Also capture at different alpha values for the p5 version
const alphaVariants = [
  { alpha: 0.2, label: "dead-zone" },
  { alpha: 2.0, label: "effective" },
  { alpha: 4.0, label: "collapse" },
];

async function capture() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-webgl", "--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  });

  for (const pg of pages) {
    const filePath = path.join(__dirname, pg.file);
    const url = `file://${filePath}`;

    console.log(`Capturing ${pg.file}...`);
    const context = await browser.newContext({
      viewport: { width: pg.width, height: pg.height },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    await page.goto(url, { waitUntil: "networkidle" });
    await page.waitForTimeout(pg.waitMs);

    const outPath = path.join(outputDir, pg.out);
    await page.screenshot({ path: outPath, fullPage: false });
    console.log(`  -> ${outPath}`);

    await context.close();
  }

  // Alpha variants for p5
  for (const variant of alphaVariants) {
    const filePath = path.join(__dirname, "superposition-p5.html");
    const url = `file://${filePath}`;
    const outName = `superposition-p5-alpha-${variant.label}.png`;

    console.log(`Capturing p5 at alpha=${variant.alpha} (${variant.label})...`);
    const context = await browser.newContext({
      viewport: { width: 800, height: 700 },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    await page.goto(url, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    // Set alpha slider
    await page.evaluate((a) => {
      const slider = document.getElementById("c-alpha");
      slider.value = a;
      slider.dispatchEvent(new Event("input"));
    }, variant.alpha);
    await page.waitForTimeout(2500);

    const outPath = path.join(outputDir, outName);
    await page.screenshot({ path: outPath, fullPage: false });
    console.log(`  -> ${outPath}`);

    await context.close();
  }

  await browser.close();
  console.log("\nDone.");
}

capture().catch((err) => {
  console.error(err);
  process.exit(1);
});
