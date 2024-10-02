/** 
 * @author Leo Vainio
 */

public class Main {
    public static void main(String[] args) {
        Lexer lexer = new Lexer();
        Parser parser = new Parser(lexer);
        new Runner(parser.getSyntaxTree());

        System.exit(0);
    }    
}