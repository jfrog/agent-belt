// (c) JFrog Ltd. (2026)

import express from "express";
import booksRouter from "./routes/books";
import authorsRouter from "./routes/authors";
import { initDb } from "./services/bookService";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/books", booksRouter);
app.use("/authors", authorsRouter);

initDb();

app.listen(PORT, () => {
  console.log(`Bookstore API listening on port ${PORT}`);
});

export default app;
