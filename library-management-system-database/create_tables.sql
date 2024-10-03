/* ----- BOOKS ----- */

CREATE TABLE IF NOT EXISTS book (
    book_id         	SERIAL,
    sequel		INT,
	
    PRIMARY KEY (book_id),
    FOREIGN KEY (sequel) REFERENCES book (book_id)
);

CREATE TABLE IF NOT EXISTS author (
    author_id		SERIAL,
    first_name          VARCHAR(50),
    last_name         	VARCHAR(50), 

    PRIMARY KEY (author_id)
);

CREATE TABLE IF NOT EXISTS book_author (
    book_id 		INT,
    author_id		INT, 

    PRIMARY KEY (book_id, author_id),
    FOREIGN KEY (book_id) REFERENCES book (book_id),
    FOREIGN KEY (author_id) REFERENCES author (author_id)
);

CREATE TABLE IF NOT EXISTS genre (
    genre_id         	SERIAL,
    genre               VARCHAR(50),

    PRIMARY KEY(genre_id)
);

CREATE TABLE IF NOT EXISTS book_genre (
    book_id 		INT,
    genre_id		INT, 

    PRIMARY KEY (book_id, genre_id),
    FOREIGN KEY (book_id) REFERENCES book (book_id),
    FOREIGN KEY (genre_id) REFERENCES genre (genre_id)
);



/* ----- RELEASE ----- */
CREATE TABLE IF NOT EXISTS publisher (
    publisher_id 	SERIAL,
    name		VARCHAR(50),

    PRIMARY KEY (publisher_id)
);

CREATE TABLE IF NOT EXISTS release (
    release_id 		SERIAL,
    book_id		INT		NOT NULL,
    isbn		VARCHAR(17),
    title		VARCHAR(255),
    language		VARCHAR(50),
    edition		VARCHAR(50),
    publisher_id	INT,
    publication_date	DATE,

    PRIMARY KEY (release_id),
    FOREIGN KEY (book_id) REFERENCES book (book_id),
    FOREIGN KEY (publisher_id) REFERENCES publisher (publisher_id)
);



/* ----- PHYSICAL BOOK ----- */

CREATE TABLE IF NOT EXISTS physical_book (
    physical_id 	SERIAL,
    release_id		INT,

    PRIMARY KEY (physical_id),
    FOREIGN KEY (release_id) REFERENCES release (release_id)
);

CREATE TABLE IF NOT EXISTS damage (
    physical_id        	INTEGER,
    damage         	VARCHAR(255),
	
    PRIMARY KEY (physical_id, damage),
    FOREIGN KEY (physical_id) REFERENCES physical_book (physical_id)
);



/* ----- USER ----- */

CREATE TABLE IF NOT EXISTS user_info (
    user_id         	SERIAL,
    email           	VARCHAR(255)    UNIQUE NOT NULL,
    first_name       	VARCHAR(50)     NOT NULL,
    last_name		VARCHAR(50)     NOT NULL,
    address         	VARCHAR(255)    NOT NULL,
	
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS admin (
    user_id         	INT		UNIQUE NOT NULL,
    phone_number    	VARCHAR(20)    	NOT NULL,
    department      	VARCHAR(50),
	
    PRIMARY KEY (user_id),
    FOREIGN KEY (user_id) REFERENCES user_info (user_id)
);

CREATE TABLE IF NOT EXISTS student (
    user_id        	INT		UNIQUE NOT NULL,
    program        	VARCHAR(50)    	NOT NULL,
	
    PRIMARY KEY (user_id),
    FOREIGN KEY (user_id) REFERENCES user_info (user_id)
);



/* ----- BORROW ----- */ 

CREATE TABLE IF NOT EXISTS borrow (
    borrow_id        	SERIAL,
    physical_id        	INT            	NOT NULL,
    user_id        	INT            	NOT NULL,
    borrow_date        	DATE            NOT NULL,
    expiration_date    	DATE            NOT NULL,
    return_date         DATE		DEFAULT NULL,
	
    PRIMARY KEY (borrow_id),
    FOREIGN KEY (physical_id) REFERENCES physical_book (physical_id),
    FOREIGN KEY (user_id) REFERENCES user_info (user_id)
);

CREATE TABLE IF NOT EXISTS fine (
    borrow_id        	INTEGER,
    fine_amount         FLOAT		NOT NULL,
	
    PRIMARY KEY (borrow_id),
    FOREIGN KEY (borrow_id) REFERENCES borrow (borrow_id)
);

CREATE TABLE IF NOT EXISTS transaction (
    transaction_id    	SERIAL,
    borrow_id        	INT            	NOT NULL,
    payment_date        DATE            NOT NULL,
    payment_method    	VARCHAR(20)    	NOT NULL,
	
    PRIMARY KEY (transaction_id),
    FOREIGN KEY (borrow_id) REFERENCES borrow (borrow_id)
);