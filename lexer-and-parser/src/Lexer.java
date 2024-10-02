/**
 * This class implements a lexer for the Leona programming language.
 * It splits up the source code of a Leona program into tokens that
 * can later on be used by the parser to create a parse tree.
 * 
 * @author Leo Vainio
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;


public class Lexer {   
    Queue<Token> tokens;
    private int lastLineNr;

    /**
     * Reads in a Leona program line by line and 
     * tokenizes the input into a queue that can 
     * later be used by the parser.
     */
    public Lexer() {
        tokens = new LinkedList<>();   

        try (
            BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        ) {

            int lineNr = 1;
            String line = null;
            while((line = stdin.readLine()) != null) {
                processLine(line, lineNr);
                lineNr++;
            }
            if(tokens.isEmpty()) System.exit(0);
            tokens.add(new Token(TokenType.EOF, lastLineNr));

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Returns the tokens created by the lexer.
     * 
     * @return A queue of tokens.
     */
    public Queue<Token> getTokens() {
        return tokens;
    }


    // process a line of input and add the tokens to list
    private void processLine(String line, int lineNr) {
        int currentIndex = 0;

        while(currentIndex < line.length()) {

            // find the next character on the line that is not whitespace: 
            while(Character.isWhitespace(line.charAt(currentIndex))) {
                currentIndex++;

                if(currentIndex >= line.length()) return;
            }

            // ----- PROCESS THE TOKEN ----- //

            Token token = null;

            // Comment, skip rest of line
            if (line.charAt(currentIndex) == '%') {
                return;
            
            // Period
            } else if (line.charAt(currentIndex) == '.') {
                token = new Token(TokenType.PERIOD, lineNr);
                currentIndex++;
            
            // Quote
            } else if (line.charAt(currentIndex) == '"') {
                token = new Token(TokenType.QUOTE, lineNr);
                currentIndex++;

            // Hex Color
            } else if (line.charAt(currentIndex) == '#') {
                if (isValidHexColor(line, currentIndex + 1)) {
                    token = new Token(TokenType.HEX, lineNr, "#" + line.substring(currentIndex + 1, currentIndex + 7));
                    currentIndex += 7;
                } else {
                    token = new Token(TokenType.ERROR, lineNr);
                }

            // Decimal
            } else if (Character.isDigit(line.charAt(currentIndex))) {
                int numDigits;
                if ((numDigits  = isValidDecimal(line, currentIndex)) > 0) {
                    int decimal = Integer.parseInt(line.substring(currentIndex, currentIndex + numDigits));
                    if(decimal <= 0) {
                        token = new Token(TokenType.ERROR, lineNr);
                    } else {
                        token = new Token(TokenType.DECIMAL, lineNr, decimal);
                        currentIndex += numDigits;
                    } 
                } else {
                    token = new Token(TokenType.ERROR, lineNr);
                }

            // Command
            } else  if (Character.isLetter(line.charAt(currentIndex))) {
                token = isValidCommand(line, currentIndex, lineNr);
                currentIndex += token.getType().toString().length();


            // Error 
            } else {
                token = new Token(TokenType.ERROR, lineNr);
            }

            lastLineNr = token.getLineNr();
            if(token.getType() == TokenType.ERROR) currentIndex++;
            tokens.add(token);
        }
    }


    // ----- Private helpers ----- //

    // Checks if a substring in line of size six 
    // is a valid hex number, starting at index.
    private boolean isValidHexColor(String line, int index) {
        for(int i = 0; i < 6; i++) {
            if(index + i < line.length()) {
                if(Character.digit(line.charAt(index + i), 16) < 0) {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    // Checks if a decimal is valid and returns the amount of
    // digits that the decimal contains. If the previous command
    // is REP this also makes sure that there is whitespace following
    // the decimal and returns the decimals size + 1. 
    // if the decimal is invalid this returns -1.
    private int isValidDecimal(String line, int index) {
        int count = 0;
        while(Character.isDigit(line.charAt(index + count))) {
            count++;
            if(index + count >= line.length()) return count; // end of line
        }
        // check for whitespace, dot or comment
        if(Character.isWhitespace(line.charAt(index+count)) || 
                                  line.charAt(index+count) == '%' || 
                                  line.charAt(index+count) == '.') {
            return count;
        }
        return -1;
    }


    // Checks to see if the token is a valid command and adds it 
    // to the queue of tokens if it is. The method returns the amount
    // of letters a command takes + 1 for whitespace. If it is invalid
    // this returns -1. 
    private Token isValidCommand(String line, int index, int lineNr) {
        int numChars = 0;
        TokenType type = null;

        // FORW
        if (index + 3 < line.length() && line.substring(index, index + 4).toLowerCase().equals("forw")) {
            numChars = 4;
            type = TokenType.FORW;

        // BACK
        } else if (index + 3 < line.length() && line.substring(index, index + 4).toLowerCase().equals("back")) {
            numChars = 4;
            type = TokenType.BACK;
        
        // LEFT
        } else if (index + 3 < line.length() && line.substring(index, index + 4).toLowerCase().equals("left")) {
            numChars = 4;
            type = TokenType.LEFT;
        
        // RIGHT
        } else if (index + 4 < line.length() && line.substring(index, index + 5).toLowerCase().equals("right")) {
            numChars = 5;
            type = TokenType.RIGHT;
        
        // DOWN
        } else if (index + 3 < line.length() && line.substring(index, index + 4).toLowerCase().equals("down")) {
            numChars = 4;
            type = TokenType.DOWN;
        
        // UP
        } else if (index + 1 < line.length() && line.substring(index, index + 2).toLowerCase().equals("up")) {
            numChars = 2;
            type = TokenType.UP;
        
        // COLOR
        } else if (index + 4 < line.length() && line.substring(index, index + 5).toLowerCase().equals("color")) {
            numChars = 5;
            type = TokenType.COLOR;

        // REP
        } else if (index + 2 < line.length() && line.substring(index, index + 3).toLowerCase().equals("rep")) {
            numChars = 3;
            type = TokenType.REP;
        
        // ERROR
        } else {
            type = TokenType.ERROR;
        }        
       
        // Make sure the command is followed by whitespace
        // add to queue of tokens in that case. UP and DOWN
        // does not need whitespace following it.
        Token token = null;
        if(!(type == TokenType.UP) && !(type == TokenType.DOWN)) {
            if(index + numChars >= line.length()) {
                token = new Token(type, lineNr);
            } else if (Character.isWhitespace(line.charAt(index+numChars))) {
                token = new Token(type, lineNr);
            } else if (line.charAt(index+numChars) == '%') {
                token = new Token(type, lineNr);
            } else {
                token = new Token(TokenType.ERROR, lineNr);
            }

        // UP and DOWN
        } else {
            token = new Token(type, lineNr);
        }

        return token;
    }
}