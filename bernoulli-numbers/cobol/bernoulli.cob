       >> SOURCE FORMAT FREE

*> Code written by Leo Vainio

IDENTIFICATION DIVISION.
PROGRAM-ID. BERNOULLI.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 BTable.
        02 Bnums COMP-2 OCCURS 25 TIMES.

    01 N PIC 9(2) VALUE 0.
    01 M PIC 9(2) VALUE 0.
    01 K PIC 9(2) VALUE 0.
    01 I PIC 9(2) VALUE 0.

    01 R  COMP-2.
    01 BM COMP-2.


PROCEDURE DIVISION.
    Receive-input.
        DISPLAY "Which Bernoulli number do you want? " WITH NO ADVANCING.
        ACCEPT N.
        COMPUTE N = N + 2.
        MOVE 1.0 TO Bnums(1).

    Bernoulli.
        PERFORM Outerloop VARYING M FROM 2 BY 1 UNTIL M=N
        COMPUTE N = N - 1.
        DISPLAY Bnums(N).
        STOP RUN.
        
    Outerloop.
        SET BM TO 0.
        PERFORM VARYING K FROM 1 BY 1 UNTIL K=M
            PERFORM Binom
            COMPUTE BM = BM - R * Bnums(K)
        END-PERFORM.
        COMPUTE BM = BM / M.
        MOVE BM TO Bnums(M).

    Binom.
        SET R TO 1.
        PERFORM VARYING I FROM 1 BY 1 UNTIL I=K
           COMPUTE R = R * (M - I + 1) / I
        END-PERFORM.
    