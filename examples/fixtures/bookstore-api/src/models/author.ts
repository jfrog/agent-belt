// (c) JFrog Ltd. (2026)

export interface Author {
  id: number;
  name: string;
  bio: string;
  born_year: number;
}

export interface CreateAuthorInput {
  name: string;
  bio: string;
  born_year: number;
}
