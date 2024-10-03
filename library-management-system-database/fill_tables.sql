/* ----- BOOKS ----- */

INSERT INTO book VALUES
/* Authors 1-20 */
(DEFAULT, 2),
(DEFAULT, 3),
(DEFAULT, 4),
(DEFAULT, 5),
(DEFAULT, 6),
(DEFAULT, 7),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL),
(DEFAULT, NULL);

INSERT INTO author VALUES
(DEFAULT, 'J.K.', 'Rowling'),
(DEFAULT, 'Kimmo', 'Eriksson'),
(DEFAULT, 'Hillevi', 'Gavel'),
(DEFAULT, 'Jon', 'Kleinberg'),
(DEFAULT, 'Eva', 'Tardos'),
(DEFAULT, 'Lars', 'Lindkvist'),
(DEFAULT, 'Jörgen', 'F Bakka'), 
(DEFAULT, 'Egil', 'Fivelsdal'),
(DEFAULT, 'Steve', 'McConnell'),
(DEFAULT, 'Robert', 'Sedgewick'),
(DEFAULT, 'Kevin', 'Wayne'),
(DEFAULT, 'Helen', 'Sharp'),
(DEFAULT, 'Jennifer', 'Preece'),
(DEFAULT, 'Rogers', 'Yvonne'),
(DEFAULT, 'Anton', 'Howard'),
(DEFAULT, 'Robert C', 'Busby'),
(DEFAULT, 'Dan', 'Brown'),
(DEFAULT, 'J.D.', 'Salinger'),
(DEFAULT, 'Scott', 'Meyers'),
(DEFAULT, 'Andreas', 'de Blanche'),
(DEFAULT, 'Marc', 'Peter Deisenroth'),
(DEFAULT, 'A. Aldo', 'Faisal'),
(DEFAULT, 'Cheng', 'Soon Ong');

INSERT INTO book_author VALUES
(1, 1),
(2, 1),
(3, 1),
(4, 1),
(5, 1),
(6, 1),
(7, 1),
(8, 2),
(8, 3),
(9, 4),
(9, 5),
(10, 6),
(10, 7),
(10, 8),
(11, 9),
(12, 10),
(12, 11),
(13, 12),
(13, 13),
(13, 14),
(14, 15),
(14, 16),
(15, 17),
(16, 17),
(17, 18),
(18, 19),
(19, 20),
(20, 21),
(20, 22),
(20, 23);

INSERT INTO genre VALUES
(DEFAULT, 'Fantasy'),
(DEFAULT, 'Matematik'),
(DEFAULT, 'Datorer & IT'),
(DEFAULT, 'Algoritmer & datastrukturer'),
(DEFAULT, 'Organisationsteori & organisationers beteende'),
(DEFAULT, 'Programmering'),
(DEFAULT, 'Datavetenskap'),
(DEFAULT, 'Människa-datorinteraktion'),
(DEFAULT, 'Thriller'),
(DEFAULT, 'Skönlitteratur'),
(DEFAULT, 'Sannolikhetskalkyl & matematisk statistik'),
(DEFAULT, 'Maskininlärning'),
(DEFAULT, 'Artificiell intelligens');
 
INSERT INTO book_genre VALUES
(1, 1),
(2, 1),
(3, 1),
(4, 1),
(5, 1),
(6, 1),
(7, 1),
(8, 2),
(9, 3),
(9, 4),
(10, 5),
(11, 3),
(11, 6),
(12, 4),
(12, 7),
(13, 8),
(14, 2),
(15, 9),
(16, 9),
(17, 10),
(18, 6),
(19, 3),
(20, 11),
(20, 12),
(20, 13);



/* ----- RELEASE ----- */

INSERT INTO publisher VALUES
(DEFAULT, 'Rabén & Sjögren'),
(DEFAULT, 'Bloomsbury Childrens Books'),
(DEFAULT, 'Studentlitteratur AB'),
(DEFAULT, 'Pearson'),
(DEFAULT, 'Liber'),
(DEFAULT, 'MICROSOFT PRESS'),
(DEFAULT, 'ADDISON-WESLEY'),
(DEFAULT, 'John Wiley & Sons Inc'),
(DEFAULT, 'Albert Bonniers Förlag'),
(DEFAULT, 'Penguin Books Ltd'),
(DEFAULT, 'O''Reilly Media, Inc, USA'),
(DEFAULT, 'Cambridge University Press');

INSERT INTO release VALUES
(DEFAULT, 1, '978-9-12-972394-6', 'Harry Potter och De vises sten', 'Svenska', '8', 1, '2019-09-16'),
(DEFAULT, 1, '978-1-40-885565-2', 'Harry Potter and the Philosopher''s Stone', 'Engelska', NULL, 2, '2014-09-01'),
(DEFAULT, 2, '978-9-12-972396-0', 'Harry Potter och Hemligheternas kammare', 'Svenska', '7', 1, '2019-11-08'),
(DEFAULT, 2, '978-1-40-885566-9', 'Harry Potter and the Chamber of Secrets', 'Engelska', NULL, 2, '2014-09-01'),
(DEFAULT, 3, '978-9-12-972395-3', 'Harry Potter och Fången från Azkaban', 'Svenska', '6', 1, '2019-09-20'),
(DEFAULT, 4, '978-9-12-972391-5', 'Harry Potter och den flammande bägaren', 'Svenska', '5', 1, '2019-10-04'),
(DEFAULT, 5, '978-9-12-972392-2', 'Harry Potter och Fenixorden', 'Svenska', '4', 1, '2019-10-04'),
(DEFAULT, 6, '978-9-12-972390-8', 'Harry Potter och halvblodsprinsen', 'Svenska', '4', 1, '2019-08-05'),
(DEFAULT, 7, '978-9-12-972393-9', 'Harry Potter och Dödsrelikerna', 'Svenska', '3', 1, '2019-11-08'),
(DEFAULT, 8, '978-9-14-408999-7', 'Diskret matematik och diskreta modeller', 'Svenska', '2', 3, '2013-07-02'),
(DEFAULT, 9, '978-1-29-202394-6', 'Algorithm Design: Pearson New International Edition', 'Engelska', '1', 4, '2013-07-30'),
(DEFAULT, 10, '978-9-14-711127-5', 'Organisationsteori : struktur, kultur, processer', 'Svenska', '6', 5, '2014-08-13'),
(DEFAULT, 11, '978-0-73-561967-8', 'Code Complete 2nd Edition', 'Engelska', '2', 6, '2004-07-01'),
(DEFAULT, 12, '978-0-32-157351-3', 'Algorithms', 'Engelska', '4', 7, '2011-03-28'),
(DEFAULT, 13, '978-1-11-954725-9', 'Interaction Design', 'Engelska', '5th edition', 8, '2019-05-10'),
(DEFAULT, 14, '978-0-47-116362-6', 'Contemporary Linear Algebra', 'Engelska', '1', 8, '2002-09-01'),
(DEFAULT, 15, '978-9-10-012319-2', 'Da Vinci-koden', 'Svenska', NULL, 9, '2009-01-27'),
(DEFAULT, 16, '978-9-10-012322-2', 'Änglar och demoner', 'Svenska', NULL, 9, '2009-01-27'),
(DEFAULT, 17, '978-0-24-195042-5', 'The Catcher in the Rye', 'Engelska', 'Reissue.', 10, '2010-03-04'),
(DEFAULT, 18, '978-1-49-190399-5', 'Effective Modern C++', 'Engelska', '1', 11, '2014-12-16'),
(DEFAULT, 19, '978-9-14-401980-2', 'Operativsystem : teori och praktiskt handhavande', 'Svenska', '1', 3, '2007-08-01'),
(DEFAULT, 20, '978-1-10-845514-5', 'Mathematics for Machine Learning', 'Engelska', NULL, 12, '2020-04-23');



/* ----- PHYSICAL BOOK ----- */

INSERT INTO physical_book VALUES
(DEFAULT, 1),
(DEFAULT, 2),
(DEFAULT, 3),
(DEFAULT, 4),
(DEFAULT, 5),
(DEFAULT, 6),
(DEFAULT, 7),
(DEFAULT, 8),
(DEFAULT, 9),
(DEFAULT, 10),
(DEFAULT, 11),
(DEFAULT, 11),
(DEFAULT, 11),
(DEFAULT, 12),
(DEFAULT, 12),
(DEFAULT, 13),
(DEFAULT, 14),
(DEFAULT, 14),
(DEFAULT, 14),
(DEFAULT, 15),
(DEFAULT, 15),
(DEFAULT, 16),
(DEFAULT, 17),
(DEFAULT, 18),
(DEFAULT, 19),
(DEFAULT, 20),
(DEFAULT, 21),
(DEFAULT, 21),
(DEFAULT, 22),
(DEFAULT, 22);

INSERT INTO damage VALUES
(5, 'Pages 66-69 ripped out'),
(11, 'Entire book water damaged'),
(14, 'Missing back cover'),
(24, 'Page 47/48 ripped'),
(29, 'Coffee stain on front cover');



/* ----- USER ----- */

INSERT INTO user_info VALUES
/* Admins 1-5 */
(DEFAULT, 'bobjohn@kth.se', 'Bob', 'Johansson',  'example street 1'),
(DEFAULT, 'annlars@kth.se', 'Anna', 'Larsson', 'example street 12B'),
(DEFAULT, 'benjbak@kth.se', 'Benjamin', 'Baker', 'example street 5'),
(DEFAULT, 'josemig@kth.se', 'Josefin', 'Miguel', 'example street 53'),
(DEFAULT, 'justbib@kth.se', 'Justin', 'Bieber', 'example street 14C'),
/* Students 6-15 */
(DEFAULT, 'natmell@kth.se', 'Natalie', 'Mellin', 'example street 55'),
(DEFAULT, 'simpan@kth.se', 'Simon', 'Lindbäck', 'example street 123B'),
(DEFAULT, 'leovain@kth.se', 'Leo', 'Vainio', 'example street 13'),
(DEFAULT, 'kaivedi@kth.se', 'Kaia', 'Vedin', 'example street 4'),
(DEFAULT, 'liseich@kth.se', 'Lisa', 'Eichler', 'example street 53'),
(DEFAULT, 'lottaag@kth.se', 'Lotta', 'Agervald', 'example street 3'),
(DEFAULT, 'tomnils@kth.se', 'Tommy', 'Nilsson', 'example street 333'),
(DEFAULT, 'kevengs@kth.se', 'Kevin', 'Engström', 'examle street 213'),
(DEFAULT, 'therese@kth.se', 'Therese', 'Tillander', 'example street 234'),
(DEFAULT, 'rayazam@kth.se', 'Raya', 'Zamil', 'example street 33');
 
INSERT INTO admin VALUES
(1, '+46 70 000 00 00', 'Department of Mathematics'),
(2, '+46 70 000 00 01', 'Department of Aeronautical and Vehicle Engineering'),
(3, '+46 70 000 00 02', 'The Department of Urban Planning and Environment'),
(4, '+46 70 000 00 03', 'Department of Mathematics'),
(5, '+46 70 000 00 04', 'Department of Philosophy and History');

INSERT INTO student VALUES
(6, 'Högskoleingenjör, medicinsk teknik'),
(7, 'Civilingenjör, datateknik'),
(8, 'Civilingenjör, öppen ingång'),
(9, 'Civilingenjör, datateknik'),
(10, 'Civilingenjör bioteknik'),
(11, 'Civilingenjör och lärare, TIKT'),
(12, 'Arkitektur'),
(13, 'Arkitektur'),
(14, 'Civilingenjör, industriell ekonomi'),
(15, 'Civilingenjör, Maskinteknik');



/* ----- BORROW ----- */ 

INSERT INTO borrow VALUES
(DEFAULT, 23, 6, '2016-09-15', '2016-09-22', '2016-09-21'),
(DEFAULT, 13, 11, '2017-05-07', '2017-05-14', '2017-05-12'),
(DEFAULT, 7, 9, '2018-12-20', '2018-12-27', '2018-12-30'),    /* late */
(DEFAULT, 27, 11, '2019-03-12', '2019-03-19', '2019-03-18'),
(DEFAULT, 2, 14, '2019-07-20', '2019-07-27', '2019-07-27'),
(DEFAULT, 9, 15, '2020-03-29', '2020-04-06', '2020-04-18'),   /* late */
(DEFAULT, 12, 7, '2020-04-07', '2020-04-14', '2020-04-30'),   /* late */
(DEFAULT, 20, 13, '2021-04-04', '2021-04-11', '2021-04-10'),
(DEFAULT, 23, 8, '2021-06-15', '2021-06-22', '2021-06-29'),   /* late */
(DEFAULT, 3, 10, '2021-11-01', '2021-11-08', '2021-11-07'),

(DEFAULT, 24, 6, '2021-10-15', '2021-10-22', NULL),           /* late */
(DEFAULT, 15, 12, '2021-10-18', '2021-10-25', NULL),          /* late */
(DEFAULT, 26, 11, '2021-10-28', '2021-11-05', NULL),          /* late */
(DEFAULT, 5, 15, '2021-11-07', '2021-11-14', NULL),
(DEFAULT, 1, 9, '2021-11-10', '2021-11-17', NULL);

INSERT INTO fine VALUES
(3, 20.0),
(6, 45.0),
(7, 30.50),
(9, 25.0),
(11, 60.99),
(12, 20.0),
(13, 20.0);

INSERT INTO transaction VALUES
(DEFAULT, 3, '2019-01-16', 'Cash'),
(DEFAULT, 6, '2020-05-01', 'Klarna'),
(DEFAULT, 7, '2020-05-15', 'Swish'),
(DEFAULT, 9, '2021-07-16', 'Card');