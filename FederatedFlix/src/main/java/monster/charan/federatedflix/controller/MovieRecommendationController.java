/*
 * The MIT License
 *
 * Copyright 2023 Charan.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package monster.charan.federatedflix.controller;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Named;
import jakarta.enterprise.context.SessionScoped;
import jakarta.inject.Inject;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import monster.charan.federatedflix.model.Movie;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.primefaces.PrimeFaces;

/**
 *
 * @author Charan
 */
@Named(value = "moviesRecController")
@SessionScoped
public class MovieRecommendationController implements Serializable {

    @Inject
    App app;

    @Inject
    SessionManager sessionManager;

    private String clientID;
    private ArrayList<String> movieRecs = new ArrayList<>();
    private ArrayList<Movie> movies = new ArrayList<>();
    private boolean loading = true;

    OkHttpClient client = new OkHttpClient().newBuilder()
            .readTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
            .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
            .build();

    @PostConstruct
    public void init() {
//        getmovies();
    }

    public void getmovies() {
        System.out.println("Get Movies called");
        loading = true;
        Request request = new Request.Builder()
                .url("http://localhost:5000/getrec/" + sessionManager.getRandomUserID())
                .get()
                .build();
        try {

            Response response = client.newCall(request).execute();

            var movieResp = response.body().string();
            System.out.println("Movies Resp:");
            System.out.println(movieResp);
            String result = movieResp.substring(1, movieResp.length() - 1);
            movieRecs = new ArrayList<String>(Arrays.asList(result.split(",")));

            for (String movieID : movieRecs) {
                movies.add(new Movie(movieID, app.getMovieData().get(movieID)));
            }

        } catch (IOException ex) {
            Logger.getLogger(App.class.getName()).log(Level.SEVERE, null, ex);
        }
        loading = false;
        System.out.println("Done");
    }

    public String getClientID() {
        return clientID;
    }

    public void setClientID(String clientID) {
        this.clientID = clientID;
    }

    public ArrayList<String> getMovieRecs() {
        return movieRecs;
    }

    public ArrayList<Movie> getMovies() {
        return movies;
    }

    public boolean isLoading() {
        return loading;
    }

    public void setLoading(boolean loading) {
        this.loading = loading;
    }

}
