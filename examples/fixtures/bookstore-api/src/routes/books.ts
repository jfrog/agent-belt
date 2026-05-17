// (c) JFrog Ltd. (2026)

import { Router, Request, Response } from "express";
import { getAllBooks, getBookById, createBook, searchBooks } from "../services/bookService";
import { validateBookInput } from "../middleware/validate";
import { parsePagination, paginate } from "../utils/pagination";

const router = Router();

router.get("/", (req: Request, res: Response) => {
  const books = getAllBooks();
  const params = parsePagination(req.query);

  if (req.query.search) {
    const results = searchBooks(req.query.search as string);
    res.json(paginate(results, params));
    return;
  }

  res.json(paginate(books, params));
});

router.get("/:id", (req: Request, res: Response) => {
  const id = parseInt(req.params.id, 10);
  const book = getBookById(id);
  if (!book) {
    res.status(404).json({ error: "Book not found" });
    return;
  }
  res.json(book);
});

router.post("/", validateBookInput, (req: Request, res: Response) => {
  const book = createBook(req.body);
  res.status(201).json(book);
});

export default router;
