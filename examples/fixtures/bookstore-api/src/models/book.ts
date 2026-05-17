// (c) JFrog Ltd. (2026)

export interface Book {
  id: number;
  title: string;
  author_id: number;
  isbn: string;
  published_year: number;
  genre: string;
}

export interface CreateBookInput {
  title: string;
  author_id: number;
  isbn: string;
  published_year: number;
  genre: string;
}
