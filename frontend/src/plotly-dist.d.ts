/**
 * `plotly.js-dist-min` ships no type declarations.
 *
 * The package is the pre-built, minified Plotly bundle, pulled in so `react-plotly.js`
 * does not drag the full un-minified library into the graph — that build triples the
 * bundle for a page that draws two figures. The trade is that the distribution has no
 * `.d.ts`, so the module is declared here.
 *
 * It is typed as the *same* `plotly.js` namespace the factory's own typings use, not as
 * `object`. Declaring it loosely would satisfy the compiler while quietly discarding every
 * guarantee `@types/react-plotly.js` provides — the figure props would still typecheck,
 * but as `any` underneath, and a misspelled trace property would fail silently at runtime
 * instead of at build time.
 */

declare module "plotly.js-dist-min" {
  import type * as Plotly from "plotly.js";

  const value: typeof Plotly;
  export default value;
}
