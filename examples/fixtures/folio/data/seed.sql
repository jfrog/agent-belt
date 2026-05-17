-- Folio bookstore seed schema + reference data.
-- (c) JFrog Ltd. (2026)
--
-- Loaded fresh on every Folio server startup so scenario assertions are
-- stable regardless of wall-clock time. Order dates are computed at
-- runtime in seed.py via the placed_days_ago column.

CREATE TABLE books (
    isbn          TEXT PRIMARY KEY,
    title         TEXT    NOT NULL,
    author        TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    year          INTEGER NOT NULL,
    price_usd     REAL    NOT NULL,
    stock_qty     INTEGER NOT NULL
);

CREATE TABLE customers (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    joined_at     TEXT NOT NULL
);

-- placed_days_ago / delivered_days_ago are populated at seed time into
-- real timestamps so the running server returns concrete dates.
CREATE TABLE orders (
    order_id              INTEGER PRIMARY KEY,
    customer_id           TEXT NOT NULL REFERENCES customers(customer_id),
    isbn                  TEXT NOT NULL REFERENCES books(isbn),
    qty                   INTEGER NOT NULL,
    unit_price_usd        REAL NOT NULL,
    status                TEXT NOT NULL,           -- placed, shipped, delivered, refunded, credited, escalated
    placed_at             TEXT NOT NULL,
    delivered_at          TEXT
);

CREATE TABLE refunds (
    refund_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    amount_usd    REAL NOT NULL,
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE store_credits (
    credit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    amount_usd    REAL NOT NULL,
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE escalations (
    ticket_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    order_id      INTEGER REFERENCES orders(order_id),
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

INSERT INTO books (isbn, title, author, category, year, price_usd, stock_qty) VALUES
  ('978-0441172719', 'Dune',                                'Frank Herbert',        'sci-fi',    1965, 18.99,  42),
  ('978-0593135204', 'Project Hail Mary',                   'Andy Weir',            'sci-fi',    2021, 19.50,  31),
  ('978-0201616224', 'The Pragmatic Programmer',            'Hunt & Thomas',        'tech',      1999, 39.99,  18),
  ('978-1449373320', 'Designing Data-Intensive Applications','Martin Kleppmann',    'tech',      2017, 49.99,  27),
  ('978-0553293357', 'Foundation',                          'Isaac Asimov',         'sci-fi',    1951, 16.00,  55),
  ('978-0812550702', 'Ender''s Game',                       'Orson Scott Card',     'sci-fi',    1985, 14.99,  64),
  ('978-0451524935', '1984',                                'George Orwell',        'classic',   1949, 12.99,  88),
  ('978-0061120084', 'To Kill a Mockingbird',               'Harper Lee',           'classic',   1960, 15.99,  47),
  ('978-0345339683', 'The Hobbit',                          'J.R.R. Tolkien',       'fantasy',   1937, 14.50, 102),
  ('978-0345418296', 'The Lord of the Rings',               'J.R.R. Tolkien',       'fantasy',   1954, 35.00,  21),
  ('978-0553380958', 'Snow Crash',                          'Neal Stephenson',      'sci-fi',    1992, 17.99,  12),
  ('978-0061767395', 'Anathem',                             'Neal Stephenson',      'sci-fi',    2008, 21.99,   8),
  ('978-0553573404', 'Hyperion',                            'Dan Simmons',          'sci-fi',    1989, 16.99,  19),
  ('978-0765377067', 'The Three-Body Problem',              'Cixin Liu',            'sci-fi',    2014, 17.00,  44),
  ('978-0316452502', 'Children of Time',                    'Adrian Tchaikovsky',   'sci-fi',    2015, 18.00,  16),
  ('978-0441478125', 'The Left Hand of Darkness',           'Ursula K. Le Guin',    'sci-fi',    1969, 14.99,   3),
  ('978-0374104092', 'Annihilation',                        'Jeff VanderMeer',      'sci-fi',    2014, 13.00,  29),
  ('978-0765377111', 'Mistborn: The Final Empire',          'Brandon Sanderson',    'fantasy',   2006, 18.99,  37),
  ('978-0765326355', 'The Way of Kings',                    'Brandon Sanderson',    'fantasy',   2010, 24.00,  22),
  ('978-1101904220', 'Recursion',                           'Blake Crouch',         'thriller',  2019, 17.00,  14),
  ('978-1101904237', 'Dark Matter',                         'Blake Crouch',         'thriller',  2016, 16.00,  25),
  ('978-0593318171', 'Klara and the Sun',                   'Kazuo Ishiguro',       'literary',  2021, 18.00,  11),
  ('978-1455563937', 'Pachinko',                            'Min Jin Lee',          'literary',  2017, 18.99,   9),
  ('978-1400033416', 'Beloved',                             'Toni Morrison',        'literary',  1987, 15.00,  17),
  ('978-0345804327', 'The Underground Railroad',            'Colson Whitehead',     'literary',  2016, 16.95,  13),
  ('978-0316126564', 'The Lies of Locke Lamora',            'Scott Lynch',          'fantasy',   2006, 17.99,  20),
  ('978-0143111597', 'Sapiens',                             'Yuval Noah Harari',    'non-fiction',2014, 22.00, 33),
  ('978-1501161933', 'Educated',                            'Tara Westover',        'memoir',    2018, 17.00,  26),
  ('978-1250301697', 'Where the Crawdads Sing',             'Delia Owens',          'literary',  2018, 18.00,  41),
  ('978-0316769174', 'The Catcher in the Rye',              'J.D. Salinger',        'classic',   1951, 13.99,  35);

INSERT INTO customers (customer_id, name, email, joined_at) VALUES
  ('C001', 'Ada Lovelace',       'ada@example.com',          '2024-03-15'),
  ('C002', 'Grace Hopper',       'grace@example.com',        '2023-11-02'),
  ('C003', 'Alan Turing',        'alan@example.com',         '2025-01-20'),
  ('C004', 'Margaret Hamilton',  'margaret@example.com',     '2024-08-10'),
  ('C005', 'ACME Books Ltd.',    'orders@acme-books.example','2023-06-01');
