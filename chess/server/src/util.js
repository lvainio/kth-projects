import { dirname, join } from "path";
import { fileURLToPath } from "url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

const resolvePath = (...path) => join(root, ...path);

export default resolvePath;
