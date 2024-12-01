IDENTIFICATION DIVISION.
PROGRAM-ID. SUM-CALCULATION.

ENVIRONMENT DIVISION.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 UserLimit    PIC 9(3) VALUE 0.
01 TotalSum     PIC 9(5) VALUE 0.
01 Counter      PIC 9(3) VALUE 0.

PROCEDURE DIVISION.
MAIN-PROCEDURE.
    DISPLAY "Enter the limit (1 to 999): ".
    ACCEPT UserLimit.
    
    IF UserLimit < 1 OR UserLimit > 999
        DISPLAY 'Error: Limit must be between 1 and 999.'
    ELSE
        PERFORM VARYING Counter FROM 1 BY 1 UNTIL Counter > UserLimit
            ADD Counter TO TotalSum
        END-PERFORM
    END-IF.
    
    DISPLAY 'The sum of numbers from 1 to ' UserLimit ' is: ' TotalSum.
    
    STOP RUN.