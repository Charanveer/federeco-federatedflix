/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/JSF/JSFManagedBean.java to edit this template
 */
package monster.charan.federatedflix.controller;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.SessionScoped;
import jakarta.inject.Inject;
import jakarta.inject.Named;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Named(value = "userMovieRatingsController")
@SessionScoped
public class UserMovieRatingsController implements Serializable {

    @Inject
    SessionManager sessionManager;

    @Inject
    App app;

    private final HashMap<String, String> movieRatings = new HashMap<>();
    private List<Map.Entry<String, String>> previousMovieRatingsList;

    @PostConstruct
    public void init() {
        var rand = String.valueOf(sessionManager.getRandomUserDataNo());
        System.out.println("Running userMovieRatingsController for ID: " + rand);

        InputStream inputStream = getClass().getResourceAsStream("/ratings.dat");

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("::");

                if (parts[0].equals(rand)) {
                    // If it is, append the value of the 3rd column (rating) to the ratings list
                    movieRatings.put(app.getMovieData().get(parts[1]), parts[2]);
                }
            }
        } catch (IOException e) {
            System.out.println("File not found: " + e.getLocalizedMessage());
        }

        System.out.println("Rated Movie: " + movieRatings);
        previousMovieRatingsList = new ArrayList<>(movieRatings.entrySet());

    }

    public SessionManager getSessionManager() {
        return sessionManager;
    }

    public List<Map.Entry<String, String>> getPreviousMovieRatingsList() {
        return previousMovieRatingsList;
    }

    public void setPreviousMovieRatingsList(List<Map.Entry<String, String>> previousMovieRatingsList) {
        this.previousMovieRatingsList = previousMovieRatingsList;
    }

}
