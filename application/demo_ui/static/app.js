import { APP_TITLE } from "./core/constants.js";
import { createAppContext } from "./core/app-context.js";
import { createAppRuntime } from "./core/app-runtime.js";

const appRoot = document.getElementById("appRoot");
const pageMode = document.body?.dataset?.page || "live";
const context = createAppContext({ appRoot, appTitle: APP_TITLE, pageMode });
const runtime = createAppRuntime(context);

window.addEventListener("beforeunload", () => {
  runtime.dispose();
});

runtime.init();
