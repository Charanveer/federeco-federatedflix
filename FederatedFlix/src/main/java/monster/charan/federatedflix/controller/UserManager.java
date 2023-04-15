/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/JSF/JSFManagedBean.java to edit this template
 */
package monster.charan.federatedflix.controller;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Named;
import jakarta.enterprise.context.ApplicationScoped;
import java.util.HashMap;

@Named(value = "userManager")
@ApplicationScoped
public class UserManager {

    private final HashMap<String, String> users = new HashMap<>();

    @PostConstruct
    public void init() {
        System.out.println("UserManager started...");
    }

    public boolean doesUserExist(String username) {
        return users.containsKey(username);
    }

    public void setUser(String username, String password) {
        users.put(username, password);
    }

    public boolean checkPassword(String username, String password) {
        var savedPass = users.get(username);

        if (savedPass == null) {
            return false;
        }

        return savedPass.equals(password);
    }

}
