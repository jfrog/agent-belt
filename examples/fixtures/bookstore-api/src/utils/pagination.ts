// (c) JFrog Ltd. (2026)

export interface PaginationParams {
  page: number;
  limit: number;
}

export interface PaginatedResult<T> {
  data: T[];
  total: number;
  page: number;
  totalPages: number;
}

export function parsePagination(query: Record<string, any>): PaginationParams {
  const page = parseInt(query.page, 10) || 1;
  const limit = Math.min(parseInt(query.limit, 10) || 10, 100);
  return { page, limit };
}

export function paginate<T>(items: T[], params: PaginationParams): PaginatedResult<T> {
  const { page, limit } = params;
  // BUG: off-by-one - page 1 should start at offset 0, but this starts at offset `limit`
  // because it uses `page` directly instead of `page - 1`
  const offset = page * limit;
  const data = items.slice(offset, offset + limit);
  const total = items.length;
  const totalPages = Math.ceil(total / limit);

  return { data, total, page, totalPages };
}
