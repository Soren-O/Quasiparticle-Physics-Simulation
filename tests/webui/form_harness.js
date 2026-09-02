/* Runs qpsim/webui/static/app.js against a minimal fake DOM so a form
   control's BINDING can be tested from Python: build the control the app
   ships for a path, feed it input the way a browser would (set .value, fire
   "change"; click buttons), and read back state.setup.

   This is not a browser. It implements exactly what renderField, renderList
   and the wizard's read-only pass touch: createElement, appendChild,
   addEventListener, dataset, value/checked/disabled, textContent, innerHTML
   (as a reset), querySelector(All) for tag/class/attribute selectors, and
   nextElementSibling. Anything else is deliberately absent, so a control
   that starts needing more fails loudly here rather than silently passing.

   Usage:  node form_harness.js <app.js> < scenario.js
   The scenario runs in the same context as app.js and must leave its result
   in `RESULT`; the harness prints JSON.stringify(RESULT). */
"use strict";
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.attributes = {};
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.textContent = "";
    this.className = "";
    this.type = "";
    this.id = "";
    this.parent = null;
    this._innerHTML = "";
  }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    return {
      add: (...names) => { el.className = [...new Set([...list(), ...names])].join(" "); },
      remove: (...names) => { el.className = list().filter((c) => !names.includes(c)).join(" "); },
      toggle: (name, force) => {
        const has = list().includes(name);
        const on = force === undefined ? !has : Boolean(force);
        if (on && !has) el.className = [...list(), name].join(" ");
        if (!on && has) el.className = list().filter((c) => c !== name).join(" ");
        return on;
      },
      contains: (name) => list().includes(name),
    };
  }
  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key === "class") this.className = String(value);
    if (key === "id") this.id = String(value);
    if (key.startsWith("data-")) {
      this.dataset[key.slice(5).replace(/-(\w)/g, (_, ch) => ch.toUpperCase())] = String(value);
    }
  }
  getAttribute(key) { return key in this.attributes ? this.attributes[key] : null; }
  addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); }
  dispatch(event) { for (const fn of this.listeners[event] || []) fn({ target: this }); }
  click() { this.dispatch("click"); }
  set innerHTML(value) { this._innerHTML = value; this.children = []; }
  get innerHTML() { return this._innerHTML; }
  *walk() { yield this; for (const c of this.children) yield* c.walk(); }
  querySelectorAll(selector) {
    return [...this.walk()].filter((el) => el !== this && matches(el, selector));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  get nextElementSibling() {
    const siblings = this.parent ? this.parent.children : [];
    return siblings[siblings.indexOf(this) + 1] || null;
  }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }
}

function matches(el, selector) {
  return selector.split(",").some((part) => {
    const m = part.trim().match(/^([a-z]*)(#[\w-]+)?((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i);
    if (!m) throw new Error(`harness: unsupported selector ${JSON.stringify(part)}`);
    const [, tag, id, classes, attrs] = m;
    if (tag && el.tagName !== tag.toUpperCase()) return false;
    if (id && el.id !== id.slice(1)) return false;
    const have = el.className.split(/\s+/).filter(Boolean);
    for (const c of classes.split(".").filter(Boolean)) if (!have.includes(c)) return false;
    for (const attr of attrs.match(/\[[^\]]+\]/g) || []) {
      const am = attr.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
      if (!am) throw new Error(`harness: unsupported attribute selector ${attr}`);
      const [, key, val] = am;
      let actual;
      if (key.startsWith("data-")) {
        actual = el.dataset[key.slice(5).replace(/-(\w)/g, (_, ch) => ch.toUpperCase())];
      } else if (key === "type") {
        actual = el.type;
      } else {
        actual = el.getAttribute(key);
      }
      if (actual === undefined || actual === null) return false;
      if (val !== undefined && actual !== val) return false;
    }
    return true;
  });
}

const document = {
  createElement: (tag) => new Element(tag),
  createElementNS: (_ns, tag) => new Element(tag),
  // Page-level lookups get a throwaway element: app.js wires its buttons at
  // load (initCopyEditing) and a null there would abort before any control
  // under test is built. Nothing reads these back.
  querySelector: () => new Element("div"),
  querySelectorAll: () => [],
  getElementById: () => new Element("div"),
  addEventListener() {},
  body: new Element("body"),
};
const noop = () => {};
const sandbox = {
  document,
  window: {
    addEventListener: noop,
    location: { hash: "", origin: "http://harness" },
    matchMedia: () => ({ matches: false, addEventListener: noop }),
    scrollTo: noop,
  },
  localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
  // An offline server, not a broken one: the app's own `api()` sees a
  // not-ok response and takes its error path, instead of an unhandled
  // rejection killing the process the moment a control asks the server.
  fetch: () => Promise.resolve({
    ok: false, status: 503,
    json: () => Promise.resolve({ ok: false, errors: ["harness: no network"] }),
  }),
  console,
  setInterval: () => 0, clearInterval: noop, setTimeout: () => 0, clearTimeout: noop,
  Image: class {}, Blob: class {}, URL: { createObjectURL: () => "" },
  navigator: {},
  RESULT: null,
};

const appSource = fs.readFileSync(process.argv[2], "utf8");
// Everything in app.js is a declaration except its top-level entry points
// (`init();`, `initCopyEditing();`), which wire the real page and reach for
// browser APIs this harness does not have. Strip those calls and nothing
// else, so every declaration stays exactly as shipped.
const entryPoints = /^[A-Za-z_$][\w$]*\(\);\s*$/gm;
const stripped = appSource.replace(entryPoints, "");
if (!/^init\(\);/m.test(appSource)) throw new Error("harness: app.js has no top-level init(); call");

vm.createContext(sandbox);
vm.runInContext(stripped, sandbox, { filename: "app.js" });
const scenario = fs.readFileSync(0, "utf8");
vm.runInContext(scenario, sandbox, { filename: "scenario.js" });
process.stdout.write(JSON.stringify(sandbox.RESULT));
