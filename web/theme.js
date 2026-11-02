/* Applies a saved light/dark choice before the page paints (loaded without defer in <head>, so there
   is no flash). No saved choice: the system setting applies through CSS. landing.js runs the toggle. */
"use strict";
(() => {
  try {
    const saved = window.localStorage.getItem("voltproof-theme");
    if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  } catch (_err) {
    // storage blocked (private mode): follow the system setting
  }
})();
