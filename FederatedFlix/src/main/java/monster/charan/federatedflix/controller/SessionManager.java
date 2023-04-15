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
import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.inject.Inject;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import java.io.IOException;
import java.io.Serializable;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

@Named(value = "sessionManager")
@SessionScoped
public class SessionManager implements Serializable {

    @Inject
    UserManager userManager;

    @Email
    @NotBlank
    private String userID;

    @NotBlank
    @Pattern(regexp = "^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$", message = "Password must contain at least one uppercase letter, one lowercase letter, one digit, one special character and no whitespace")
    private String password;

    private String randomUserID;
    private int randomUserDataNo;

    private boolean loggedIn = false;

    @PostConstruct
    public void init() {
        System.out.println("Session started");
    }

    /**
     * Attempts to login the user with the given userID and password.
     *
     * @return A string representing the outcome of the login attempt.
     *
     * If the login is successful, the user is redirected to the home page. If
     * the userID is invalid, the user is redirected to the index page with an
     * error message indicating that the user does not exist. If the password is
     * incorrect, the user is redirected to the index page with an error message
     * indicating that the password is incorrect. This method also generates a
     * random userID if the login is successful and sets the loggedIn property
     * to true.
     *
     */
    public String login() {
        System.out.println("Logging in user: " + userID);
        if (!userManager.doesUserExist(userID)) {
            FacesContext.getCurrentInstance().addMessage("auth-form", new FacesMessage(FacesMessage.SEVERITY_ERROR, "Error", "User does not exist."));
            return "index";
        }

        if (userManager.checkPassword(userID, password)) {
            loggedIn = true;

            // create a new Random object
            Random rand = new Random();

            // generate a random integer between 0 and 6039
            int randomNum = rand.nextInt(6040);
            randomUserDataNo = randomNum + 1;

            // convert the random integer to a string
            randomUserID = String.valueOf(randomNum);

            return "home?faces-redirect=true";
        } else {
            FacesContext.getCurrentInstance().addMessage("auth-form", new FacesMessage(FacesMessage.SEVERITY_ERROR, "Error", "Password incorrect."));
            return "index";
        }

    }

    /**
     *
     * @return
     */
    public String register() {
        System.out.println("Registering in user: " + userID);

        if (userManager.doesUserExist(userID)) {
            FacesContext.getCurrentInstance().addMessage("auth-form", new FacesMessage(FacesMessage.SEVERITY_ERROR, "Error", "User already registered."));
            return "index";
        }

        // create a new Random object
        Random rand = new Random();

        // generate a random integer between 0 and 6039
        int randomNum = rand.nextInt(6040);
        randomUserDataNo = randomNum + 1;

        // convert the random integer to a string
        randomUserID = String.valueOf(randomNum);

        userManager.setUser(userID, password);
        return "home?faces-redirect=true";
    }

    /**
     *
     */
    public void signOut() {
        System.out.println("Signing Out");
        FacesContext.getCurrentInstance().getExternalContext().invalidateSession();
        try {
            FacesContext.getCurrentInstance().getExternalContext().redirect("index");
        } catch (IOException ex) {
            Logger.getLogger(SessionManager.class.getName()).log(Level.SEVERE, null, ex);
        }
        //        return "/index?faces-redirect=true";
    }

    public String getUserID() {
        return userID;
    }

    public void setUserID(String userID) {
        this.userID = userID;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public boolean isLoggedIn() {
        return loggedIn;
    }

    public String getRandomUserID() {
        return randomUserID;
    }

    public void setRandomUserID(String randomUserID) {
        this.randomUserID = randomUserID;
    }

    public int getRandomUserDataNo() {
        return randomUserDataNo;
    }

}
