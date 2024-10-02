/**
 * This class creates a syntax tree from the tokens
 * generated by the lexer. 
 * 
 * @author Leo Vainio
 */

import java.util.ArrayList;
import java.util.Queue;

public class Parser {
    Queue<Token> tokens;
    ParseTree syntaxTree;

    /**
     * Start parsing the tokens and create a syntax tree.
     */
    public Parser(Lexer lexer) {
        tokens = lexer.getTokens();
        syntaxTree = null;

        parse();
    }

    /**
     * @return the parse tree created in the parsing process.
     */
    public ParseTree getSyntaxTree() {
        return syntaxTree;
    }

    /**
     * Reads tokens created by lexer and creates a parsetree
     * that can later be executed in Execute.java
     */
    private void parse() {  
        if(!tokens.isEmpty()) {
            syntaxTree = getInstruction();
        }
        ParseTree currentNode = syntaxTree;   
        while(!tokens.isEmpty()) {
            currentNode.setNext(getInstruction());
            if(currentNode.getNext() != null) {
                currentNode = currentNode.getNext();
            }
        }
    }

    /**
     * This method returns an instruction in the form of a parsetree or
     * throws a syntaxErrorException if the instruction is invalid.
     * 
     * @param type Instruction type
     * @return A parsetree
     */
    private ParseTree getInstruction() {
        Token token = tokens.peek();
    
        switch(token.getType()) {
            case UP:
            case DOWN:  
                return upDownInstruction();

            case FORW: 
            case BACK:
            case LEFT:
            case RIGHT: 
                return forwBackLeftRightInstruction();

            case COLOR: 
                return colorInstruction();
            
            case REP:  
                return repInstruction();

            case EOF:
                tokens.remove();
                break;

            default:    
                syntaxError(token.getLineNr());
                break;
        }
        return null;
    }

    // Returns a parsetree representing an UP or a DOWN instruction.
    private ParseTree upDownInstruction() {
        Token upDown = tokens.poll();
        if(tokens.isEmpty()) syntaxError(upDown.getLineNr());
        Token period = tokens.poll();
        if(period.getType() != TokenType.PERIOD) {
            syntaxError(period.getLineNr());
        }
        return new PenNode(upDown.getType());
    }

     // Returns a parsetree representing a FORW, BACK, LEFT or RIGHT instruction.
     private ParseTree forwBackLeftRightInstruction() {
        Token instruction = tokens.poll();
        if(tokens.isEmpty()) syntaxError(instruction.getLineNr());
        Token decimal = tokens.poll();
        if(tokens.isEmpty()) syntaxError(decimal.getLineNr());

        if(decimal.getType() == TokenType.DECIMAL) {
            Token period = tokens.poll();
            if(period.getType() != TokenType.PERIOD) {
                syntaxError(period.getLineNr());
            } 
        } else {
            syntaxError(decimal.getLineNr());
        }
        return new MoveNode(instruction.getType(), (int) decimal.getArgument());
    }

    // Returns a parsetree representing a COLOR instruction. 
    private ParseTree colorInstruction() {
        Token color = tokens.poll();
        if(tokens.isEmpty()) syntaxError(color.getLineNr());
        Token hex = tokens.poll();
        if(tokens.isEmpty()) syntaxError(hex.getLineNr());

        if(hex.getType() == TokenType.HEX) {
            Token period = tokens.poll();
            if(period.getType() != TokenType.PERIOD) {
                syntaxError(period.getLineNr());
            } 
        } else {
            syntaxError(hex.getLineNr());
        }
        return new ColorNode((String) hex.getArgument());
    }


    // Check if a REP-instruction is syntactically correct and return its parse tree.
    private ParseTree repInstruction() {
        Token rep = tokens.poll(); // remove rep token
        if(tokens.isEmpty()) syntaxError(rep.getLineNr());
        Token decimal = tokens.poll();
        if(tokens.isEmpty()) syntaxError(decimal.getLineNr());

        int reps = 0;
        ArrayList<ParseTree> instructions = new ArrayList<>();
        if(decimal.getType() == TokenType.DECIMAL) {
            reps = (int) decimal.getArgument();

            // NO QUOTE
            if(tokens.peek().getType() != TokenType.QUOTE) {
                instructions.add(getInstruction());
            
            // QUOTE
            } else {
                Token quote = tokens.poll(); // remove first quote
                if(tokens.isEmpty()) syntaxError(quote.getLineNr());
                while(tokens.peek().getType() != TokenType.QUOTE) {
                    if(tokens.peek().getType() == TokenType.EOF) syntaxError(tokens.peek().getLineNr()); 
                    instructions.add(getInstruction());
                }
                if(instructions.isEmpty()) syntaxError(tokens.peek().getLineNr());
                tokens.remove(); // remove second quote 
            }

        // No DECIMAL after REP 
        } else {
            syntaxError(decimal.getLineNr());
        } 

        return new RepNode(reps, instructions);
    }

    // Throws a syntax error and specifies which line it occurred on.
    private void syntaxError(int lineNr) {
        try {
            throw new SyntaxErrorException(lineNr);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }
    }
}