import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import { App } from "./App";
import { ThemeProvider } from "./theme";

import "./styles/tokens.css";
import "./styles/global.css";
import "./styles/layout.css";
import "./styles/data.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("Root container #root is missing from index.html");
}

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>,
);
