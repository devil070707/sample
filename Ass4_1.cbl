IDENTIFICATION DIVISION.
PROGRAM-ID. AgeCheck.

ENVIRONMENT DIVISION.
INPUT-OUTPUT SECTION.

DATA DIVISION.
WORKING-STORAGE SECTION.
01 AGE      PIC 99.
01 CATEGORY PIC A(20).

PROCEDURE DIVISION.
MAIN-PARA.
    DISPLAY "Enter your age: " WITH NO ADVANCING.
    ACCEPT AGE.
    
    IF AGE >= 18
        MOVE "Adult" TO CATEGORY
    ELSE
        MOVE "Minor" TO CATEGORY
    END-IF.
    
    DISPLAY "You are an " CATEGORY.
    
    STOP RUN.