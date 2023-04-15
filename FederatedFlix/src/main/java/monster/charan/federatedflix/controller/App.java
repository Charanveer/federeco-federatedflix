package monster.charan.federatedflix.controller;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Named;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

/**
 * App controller to share state across all application users.
 *
 */
@Named(value = "app")
@ApplicationScoped
public class App {

    private final HashMap<String, String> movieData = new HashMap<>();

    @PostConstruct
    public void init() {
//        File file = new File("C:\\Users\\Charan\\Downloads\\ml-1m\\ml-1m\\movies.dat");
//        try {
//            Scanner scanner = new Scanner(file);
//
//            // Read each line of the file
//            while (scanner.hasNextLine()) {
//                String line = scanner.nextLine();
//
//                // Split the line into key and value using '::' as the separator
//                String[] parts = line.split("::");
//                String key = parts[0];
//                String value = parts[1];
//
//                movieData.put(key, value);
//
//                // Print the key-value pair
//                System.out.println(key + " -> " + value);
//            }
//
//            scanner.close();
//        } catch (FileNotFoundException e) {
//            System.out.println("File not found: " + file.getPath());
//        }

        InputStream inputStream = getClass().getResourceAsStream("/movies.dat");

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Split the line into key and value using '::' as the separator
                String[] parts = line.split("::");
                String key = parts[0];
                String value = parts[1];

                movieData.put(key, value);

                // Print the key-value pair
//                System.out.println(key + " -> " + value);
            }
        } catch (IOException e) {
            System.out.println("File not found: " + e.getLocalizedMessage());
        }

    }

    public HashMap<String, String> getMovieData() {
        return movieData;
    }

}
