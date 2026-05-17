// (c) JFrog Ltd. (2026)

import { initDb } from "../src/services/bookService";

beforeAll(() => {
  initDb(":memory:");
});
