/**
 * @author Leo Vainio
 */

// Simple class to store and get information from a token.
public class Token {
    private TokenType type;
    private int lineNr;
    private Object argument; 

    // Used for: FORW, BACK, LEFT, RIGHT, DOWN, UP, COLOR, REP, PERIOD, QUOTE, ERROR, EOF
    public Token(TokenType type, int lineNr) {
        this.type = type;
        this.lineNr = lineNr;
        this.argument = null;
    }

    // Used for: DECIMAL, HEX
    public Token(TokenType type, int lineNr, Object argument) {
        this.type = type;
        this.lineNr = lineNr;
        this.argument = argument;
    }

    public TokenType getType() {
        return type;
    }

    public int getLineNr() {
        return lineNr;
    }

    public Object getArgument() {
        return argument;
    }      
}