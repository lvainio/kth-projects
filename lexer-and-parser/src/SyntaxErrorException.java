/**
 * @author Leo Vainio
 */

public class SyntaxErrorException extends Exception {

    /**
     * This is thrown when the parser finds a syntax error,
     * prints out an error message specifying which line it 
     * occurred on.
     * 
     * @param lineNr The line that contains the syntax error
     */
    public SyntaxErrorException(int lineNr) {
        super("Syntaxfel på rad " + lineNr);
    }
}