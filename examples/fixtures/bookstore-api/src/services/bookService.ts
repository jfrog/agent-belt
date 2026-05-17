// (c) JFrog Ltd. (2026)

import Database from "better-sqlite3";
import { Book, CreateBookInput } from "../models/book";

let db: Database.Database;

export function initDb(dbPath: string = ":memory:"): void {
  db = new Database(dbPath);
  db.exec(`
    CREATE TABLE IF NOT EXISTS books (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      author_id INTEGER NOT NULL,
      isbn TEXT UNIQUE NOT NULL,
      published_year INTEGER NOT NULL,
      genre TEXT NOT NULL
    )
  `);
}

export function getAllBooks(): Book[] {
  return db.prepare("SELECT * FROM books").all() as Book[];
}

export function getBookById(id: number): Book | undefined {
  return db.prepare("SELECT * FROM books WHERE id = ?").get(id) as Book | undefined;
}

export function searchBooks(title: string): Book[] {
  // BUG: SQL injection - string interpolation instead of parameterized query
  const query = `SELECT * FROM books WHERE title LIKE '%${title}%'`;
  return db.prepare(query).all() as Book[];
}

export function createBook(input: CreateBookInput): Book {
  const stmt = db.prepare(
    "INSERT INTO books (title, author_id, isbn, published_year, genre) VALUES (?, ?, ?, ?, ?)"
  );
  const result = stmt.run(input.title, input.author_id, input.isbn, input.published_year, input.genre);
  return { id: result.lastInsertRowid as number, ...input };
}

export function deleteBook(id: number): boolean {
  const result = db.prepare("DELETE FROM books WHERE id = ?").run(id);
  return result.changes > 0;
}
