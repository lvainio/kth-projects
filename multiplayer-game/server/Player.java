/**
 * This class is used to store information about players in the game, 
 * including their position, health, damage.
 * 
 * @author Leo Vainio
 */

public class Player {
    private int xPos;
    private int yPos;
    private int hp;
    private int dp;

    /**
     * Constructor for a Player object. Sets the position specified 
     * in parameters and sets the standard health and damage.
     * 
     * @param x x-position of the player
     * @param y y-position of the player
     */
    public Player(int x, int y) {
        xPos = x;
        yPos = y; 
        hp = 20;
        dp = 10;
    }

    /**
     * This method moves the player 1 step in the direction that is
     * specified in the parameter.
     * 
     * @param dir Direction in which the player should move.
     */
    public void move(Direction dir) {
        switch (dir) {
            case UP:
                yPos--;
                break;
            case DOWN:
                yPos++;
                break;
            case LEFT:
                xPos--;
                break;
            case RIGHT:
                xPos++;
                break;
            default:
                System.err.println("Invalid direction!");
                break;
        }
    }

    /**
     * @param amount of health to add or remove (negative value).
     */
    public void addHP(int amount) {
        hp += amount;
    }

    /**
     * @param amount of damage points to add to player.
     */
    public void addDP(int amount) {
        dp += amount;
    }

    /**
     * @return the x-position of the player.
     */
    public int getXPos() {
        return xPos;
    }

    /**
     * @return the y-position of the player.
     */
    public int getYPos() {
        return yPos;
    }

    /**
     * @return the health of the player.
     */
    public int getHP() {
        return hp;
    }
    
    /**
     * @return the damage points of the player.
     */
    public int getDP() {
        return dp;
    }
}
