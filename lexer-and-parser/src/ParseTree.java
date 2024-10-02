/**
 * Parse tree for the Leona language.
 * 
 * @author Leo Vainio
 */

import java.util.List;

// Simple tree datastructure
abstract class ParseTree {
    private ParseTree next;
    protected TokenType type;

    public void setNext(ParseTree p) {
        this.next = p;
    }

    public ParseTree getNext() {
        return next;
    }

    public TokenType getType() {
        return type;
    }
}



// Used for: UP, DOWN
class PenNode extends ParseTree {
    public PenNode(TokenType type) {
        this.type = type;
    }
}



// Used for: FORW, BACK, LEFT, RIGHT
class MoveNode extends ParseTree {
    private int argument;

    public MoveNode(TokenType type, int argument) {
        this.type = type;
        this.argument = argument;
    }

    public int getArgument() {
        return argument;
    }
}



// Used for: COLOR
class ColorNode extends ParseTree {
    private String color;

    public ColorNode(String color) {
        this.type = TokenType.COLOR;
        this.color = color;
    }

    public String getColor() {
        return color;
    }
}



// Used for: REP
class RepNode extends ParseTree {
    private List<ParseTree> instructions;
    private int reps;

    public RepNode(int reps, List<ParseTree> instructions) {
        this.type = TokenType.REP;
        this.instructions = instructions;
        this.reps = reps;
    }

    public List<ParseTree> getInstructions() {
        return instructions;
    }

    public int getReps() {
        return reps;
    }
}