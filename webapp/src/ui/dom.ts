// Minimal hyperscript-style DOM builder -- no virtual DOM, no diffing.
// Views in this app re-render by rebuilding a subtree and swapping it in
// (optionally through a View Transition), which is simple to reason about
// at this app's scale and keeps the dependency count at zero.

type Child = Node | string | number | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, unknown> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = String(value);
    else if (key === "html") el.innerHTML = String(value);
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
    } else if (key in el) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (el as any)[key] = value;
    } else {
      el.setAttribute(key, String(value));
    }
  }
  for (const child of children.flat(Infinity as 1)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** Same as h(), but for SVG elements (createElementNS) -- circle, path, defs, etc. */
export function hs(tag: string, props: Record<string, unknown> = {}, ...children: Child[]): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
    } else {
      el.setAttribute(key, String(value));
    }
  }
  for (const child of children.flat(Infinity as 1)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function mount(root: HTMLElement, node: Node) {
  root.replaceChildren(node);
}

/** Swaps the app root's content through a View Transition when available, falling back to a direct swap. */
export function transitionMount(root: HTMLElement, build: () => Node) {
  const doc = document as Document & { startViewTransition?: (cb: () => void) => void };
  if (typeof doc.startViewTransition === "function") {
    doc.startViewTransition(() => mount(root, build()));
  } else {
    mount(root, build());
  }
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

export function highlight(text: string, query: string): string {
  const escaped = escapeHtml(text);
  const terms = query.trim().split(/\s+/).filter((t) => t.length > 1).map((t) => t.toLowerCase());
  if (terms.length === 0) return escaped;
  const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return escaped.replace(pattern, "<mark>$1</mark>");
}
