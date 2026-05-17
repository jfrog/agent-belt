// (c) JFrog Ltd. (2026)

import { Request, Response, NextFunction } from "express";

export function validateBookInput(req: Request, res: Response, next: NextFunction): void {
  const { title, author_id, isbn, published_year, genre } = req.body;

  const errors: string[] = [];
  if (!title || typeof title !== "string") errors.push("title is required");
  if (!author_id || typeof author_id !== "number") errors.push("author_id is required");
  if (!isbn || typeof isbn !== "string") errors.push("isbn is required");
  if (!published_year || typeof published_year !== "number") errors.push("published_year is required");
  if (!genre || typeof genre !== "string") errors.push("genre is required");

  if (errors.length > 0) {
    res.status(400).json({ errors });
    return;
  }
  next();
}
