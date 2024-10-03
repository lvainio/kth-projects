/**
 * Server class for a simple game application. Can handle 
 * multiple clients at the same time. 
 * 
 * @author Leo Vainio
 */

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class Server {

    /**
     * Constructor for the Server. Initiates the server and keeps
     * listening for new clients. If a client connects a new thread
     * gets created to handle that specific client.
     * 
     * @param port The port that the ServerSocket get bound to
     */
    public Server(int port) {
        try (
            ServerSocket serverSocket = new ServerSocket(port)
        ) {
            Game game = new Game();
            Socket clientSocket;
            while((clientSocket = serverSocket.accept()) != null) {
                System.out.println("A client connected to the server");
                new ServerThread(game, clientSocket).start();
            }
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Initiates the server with a ServerSocket bound to a 
     * specified port number.
     * 
     * @param args The port that the ServerSocket get bound to
     */
    public static void main(String[] args) {
        if(args.length != 1) {
            System.err.println("Usage: java Server <port number>");
            System.exit(1);
        }
        int port = Integer.parseInt(args[0]);
        new Server(port);
    }
}