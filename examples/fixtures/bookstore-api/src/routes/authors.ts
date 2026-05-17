// (c) JFrog Ltd. (2026)

import { Router, Request, Response } from "express";

const authors = [
  { id: 1, name: "J.K. Rowling", bio: "British author", born_year: 1965 },
  { id: 2, name: "George Orwell", bio: "English novelist", born_year: 1903 },
];

const router = Router();

// BUG: no error handling - crashes on malformed requests, no try/catch
// BUG: no input validation on POST
// BUG: no tests for any of these endpoints

router.get("/", (req: Request, res: Response) => {
  res.json(authors);
});

router.get("/:id", (req: Request, res: Response) => {
  const id = parseInt(req.params.id, 10);
  const author = authors.find((a) => a.id === id);
  if (!author) {
    res.status(404).json({ error: "Author not found" });
    return;
  }
  res.json(author);
});

router.post("/", (req: Request, res: Response) => {
  // No input validation - accepts anything
  const newAuthor = {
    id: authors.length + 1,
    ...req.body,
  };
  authors.push(newAuthor);
  res.status(201).json(newAuthor);
});

export default router;
