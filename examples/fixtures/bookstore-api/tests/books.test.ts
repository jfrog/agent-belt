// (c) JFrog Ltd. (2026)

import { getAllBooks, createBook, getBookById, searchBooks } from "../src/services/bookService";
import { initDb } from "../src/services/bookService";

describe("Book Service", () => {
  beforeEach(() => {
    initDb(":memory:");
  });

  it("should return empty list initially", () => {
    expect(getAllBooks()).toEqual([]);
  });

  it("should create and retrieve a book", () => {
    const book = createBook({
      title: "Test Book",
      author_id: 1,
      isbn: "978-0-000-00000-0",
      published_year: 2024,
      genre: "Fiction",
    });
    expect(book.id).toBe(1);
    expect(book.title).toBe("Test Book");

    const retrieved = getBookById(1);
    expect(retrieved).toBeDefined();
    expect(retrieved!.title).toBe("Test Book");
  });

  it("should return undefined for non-existent book", () => {
    expect(getBookById(999)).toBeUndefined();
  });
});
