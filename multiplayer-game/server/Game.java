/**
 * This class stores information about the game such 
 * as the map, players and so on.
 * 
 * @author Leo Vainio
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Game {
    // Map
    private final int ROWS = 20;
    private final int COLS = 100;
    private final byte EMPTY = ' ';
    private final byte WALL = '#';
    private final byte PLAYER = '@';
    private final byte HEALTH = 'H';
    private final byte SWORD = 'S';
    private byte[][] map;

    // Players
    private ArrayList<Player> players;
    private int playerCount;

    // Game state
    private enum State {
        GameNotStarted,
        Game, 
        GameOver
    }
    private State state;

    /**
     * Constructor for Game class.
     */
    public Game() {
        initMap();
        players = new ArrayList<>();
        playerCount = 0;
        state = State.GameNotStarted;
    }

    /**
     * Takes the input from the client and makes changes to the 
     * game dependent on the input. For example moving the player
     * a direction. 
     * 
     * @param input Input from client
     * @param player The clients player in-game
     */
    public void processInput(String input, Player player) {
        if(player.getHP() <= 0) return;
        if(state == State.GameNotStarted) return;
        if(state == State.GameOver) {
            String gameOver = "GAME  OVER";
            int row = 17;
            int col = 45;
            for(byte b : gameOver.getBytes()) {
                map[row][col] = b;
                col++;
            }
            return;
        }

        switch(input) {
            case "UP":
                movePlayer(player, Direction.UP, player.getXPos(), player.getYPos()-1);
                break;
            case "DOWN":
                movePlayer(player, Direction.DOWN, player.getXPos(), player.getYPos()+1);
                break;
            case "LEFT":
                movePlayer(player, Direction.LEFT, player.getXPos()-1, player.getYPos());
                break;
            case "RIGHT":
                movePlayer(player, Direction.RIGHT, player.getXPos()+1, player.getYPos());
                break;
            case "STILL":
                break; 
            default: 
                System.err.println("Error: invalid message");
        }
    }

    /**
     * Adds a new player to the game on a randomized
     * position on the map. 
     * 
     * @return the new player
     */
    public Player addPlayer() {
        Random rng = new Random();
        Player player = null;
        while(player == null) {
            int x = rng.nextInt(COLS);
            int y = rng.nextInt(ROWS);
            if(map[y][x] == ' ') {
                player = new Player(x, y);
                players.add(player);
                map[y][x] = PLAYER;
            }
        }
        playerCount++;
        if(playerCount >= 2) {
            state = State.Game;
        }
        return player;
    }

    /**
     * Removes the specified player from the game.
     * 
     * @param player Player to be removed
     */
    public void removePlayer(Player player) {
        map[player.getYPos()][player.getXPos()] = ' ';
        players.remove(player);
        playerCount--;
        if(state == State.Game && playerCount < 2) {
            state = State.GameOver;
        }
    }

    /**
     * Returns the current state of the map.
     * 
     * @return The map
     */
    public byte[][] getMap() {
        return map;
    }

    /**
     * Reads in map from a txt file. 
     */
    private void initMap() {
        try (
            BufferedReader fileReader = new BufferedReader(new FileReader("map.txt"));
        ) {
            map = new byte[ROWS][COLS];
            String line;
            int lineNr = 0;
            while((line = fileReader.readLine()) != null) {
                map[lineNr] = line.getBytes();
                lineNr++;
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Randomize health items and sword pick-ups
        Random rng = new Random();
        for(int i = 0; i < 10; i++) {
            int row = 0; 
            int col = 0;
            while(map[row][col] != ' ') {
                row = rng.nextInt(ROWS);
                col = rng.nextInt(COLS);  
            }
            map[row][col] = SWORD;

            row = 0;
            col = 0;
            while(map[row][col] != ' ') {
                row = rng.nextInt(ROWS);
                col = rng.nextInt(COLS);  
            }
            map[row][col] = HEALTH;
        }
    }

     /**
     * Checks for collisions with walls, players and items when a player moves
     */
    private void movePlayer(Player player, Direction dir, int newX, int newY) {
        if(map[newY][newX] == EMPTY) {
            map[player.getYPos()][player.getXPos()] = EMPTY;
            map[newY][newX] = PLAYER;
            player.move(dir);

        } else if (map[newY][newX] == WALL) {
            return;
            
        } else if (map[newY][newX] == PLAYER) {
            Player opponent = findPlayerByPosition(newX, newY);
            if(opponent != null) {
                opponent.addHP(-player.getDP());
                if(opponent.getHP() <= 0) {
                    removePlayer(opponent);
                }
            }
    
        } else if (map[newY][newX] == HEALTH) {
            map[player.getYPos()][player.getXPos()] = EMPTY;
            map[newY][newX] = PLAYER;
            player.addHP(10);
            player.move(dir);

        } else if (map[newY][newX] == SWORD) {
            map[player.getYPos()][player.getXPos()] = EMPTY;
            map[newY][newX] = PLAYER;
            player.addDP(5);
            player.move(dir);

        } else {
            System.err.println("Error: invalid map character");
        }
    }

    /**
     * Returns a player in the specified position. null if no
     * player exist in that position.
     */
    private Player findPlayerByPosition(int x, int y) {
        for(Player p : players) {
            if(p.getXPos() == x && p.getYPos() == y) {
                return p;
            }
        }
        return null;
    }
}
