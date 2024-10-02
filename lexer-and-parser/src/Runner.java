/**
 * This class executes a Leona program from a parsetree
 * created by the parser.
 * 
 * @author Leo Vainio
 */

import java.util.List;

public class Runner {
    private String color;
    private double x;
    private double y;
    private double dir;
    private boolean isPenDown;

    /**
     * Initialize Leona in start position and begin
     * executing the program created by the parser.
     * 
     * @param parseTree
     */
    public Runner(ParseTree syntaxTree) {
        this.color = "#0000FF";
        this.x = 0.0;
        this.y = 0.0;
        this.dir = 0.0;
        this.isPenDown = false;

        ParseTree curr = syntaxTree;
        while(curr != null) {
            runInstruction(curr);
            curr = curr.getNext();
        }
    }

    /**
     * Executes one instruction in the Leona language.
     * 
     * @param instruction
     */
    private void runInstruction(ParseTree instruction) {
        TokenType type = instruction.getType();
        switch(type) {
            case UP:
                setIsPenDown(false);
                break;
            case DOWN:
                setIsPenDown(true);
                break;
            case FORW:
                move(((MoveNode) instruction).getArgument());
                break;
            case BACK:
                move(-((MoveNode) instruction).getArgument());
                break;
            case LEFT:
                turnLeona(((MoveNode) instruction).getArgument());
                break;
            case RIGHT:
                turnLeona(-((MoveNode) instruction).getArgument());
                break;
            case COLOR:
                setColor(((ColorNode) instruction).getColor());
                break;
            case REP:
                runRepInstruction((RepNode) instruction);
            default:
                System.err.println("Invalid instruction");
                break;
        }
    }

    /**
     * Executes a rep instruction
     * 
     * @param instruction
     */
    private void runRepInstruction(RepNode instruction) {
        int reps = instruction.getReps();
        List<ParseTree> instructions = instruction.getInstructions();

        for(int i = 0; i < reps; i++) {
            for(ParseTree p : instructions) {
                runInstruction(p);
            }
        }
    }

    /**
     * Moves Leona a specified distance
     * in the direction she is currently facing.
     * If the pen is down it will also draw a line.
     * 
     * @param distance
     */
    private void move(int distance) {
        double prevX = x;
        double prevY = y;

        x = x + distance * Math.cos(Math.PI * dir/180);
        y = y + distance * Math.sin(Math.PI * dir/180);

        if(isPenDown == true) {
            draw(prevX, prevY);
        }
    }
    
    /**
     * Turn Leona x degrees to the left,
     * a negative argument is equivalent with
     * turning Leona to the right.
     * 
     * @param degree Amount of degrees Leona will turn.
     */
    private void turnLeona(int degree) {
        this.dir += degree;
    }

    // Raise or lower the pen
    private void setIsPenDown(boolean b) {
        this.isPenDown = b;
    }

    // Set the color of the pen
    private void setColor(String color) {
        this.color = color;
    }

    /**
     * Prints out one line segment
     * 
     * @param x Previous x position
     * @param y Previous y position
     */
    private void draw(double x, double y) {
        System.out.println(color + " " +
                            String.format("%.4f", x) + " " +
                            String.format("%.4f", y) + " " +
                            String.format("%.4f", this.x) + " " +
                            String.format("%.4f", this.y)
        );
    }
}