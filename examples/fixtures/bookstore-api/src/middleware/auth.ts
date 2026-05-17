// (c) JFrog Ltd. (2026)

import { Request, Response, NextFunction } from "express";

interface TokenPayload {
  sub: string;
  role: string;
  exp: number;
}

function decodeToken(token: string): TokenPayload | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const payload = JSON.parse(Buffer.from(parts[1], "base64url").toString());
    return payload as TokenPayload;
  } catch {
    return null;
  }
}

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    res.status(401).json({ error: "Missing authorization header" });
    return;
  }

  const token = header.slice(7);
  const payload = decodeToken(token);

  if (!payload) {
    res.status(401).json({ error: "Invalid token" });
    return;
  }

  // BUG: doesn't validate token expiry - expired tokens are accepted
  // Should check: if (payload.exp < Date.now() / 1000) { return 401 }

  (req as any).user = payload;
  next();
}

export function requireRole(role: string) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as any).user;
    if (!user || user.role !== role) {
      res.status(403).json({ error: "Forbidden" });
      return;
    }
    next();
  };
}
