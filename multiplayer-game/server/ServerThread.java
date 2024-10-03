/**
 * A server class that handles a single client
 * that is connected to the server.
 * 
 * @author Leo Vainio
 */

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;
import java.net.SocketException;

import Game;
import Player;

public class ServerThread extends Thread {
    private Game game; 
    private Player player; 
    private Socket clientSocket;
    
    /**
     * Constructor for ServerThread. Set fields are later used
     * in run method.
     * 
     * @param game Game object
     * @param clientSocket Socket bound to client
     */
    public ServerThread(Game game, Socket clientSocket) {
        this.game = game;
        this.player = game.addPlayer();
        this.clientSocket = clientSocket;
    }

    /**
     * Send graphics to client and listen to messages from
     * client following the protocol.
     */
    public void run() {
        try (
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            DataOutputStream out = new DataOutputStream(clientSocket.getOutputStream())
        ) {
            for(byte[] line : game.getMap()) {
                out.writeBytes(new String(line));
            }
            String clientMessage;
            while((clientMessage = in.readLine()) != null) {
                game.processInput(clientMessage.trim(), player);
                for(byte[] line : game.getMap()) {
                    out.writeBytes(new String(line));
                }
                Thread.sleep(10);
            }            
        } catch(SocketException e) {
            e.printStackTrace();
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(1);
        } catch (InterruptedException e) {
            e.printStackTrace();
            System.exit(1);
        }  finally {
            game.removePlayer(player);
            System.out.println("A client disconnected from the server");
            try {
                if(!clientSocket.isClosed()) {
                    clientSocket.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
}
