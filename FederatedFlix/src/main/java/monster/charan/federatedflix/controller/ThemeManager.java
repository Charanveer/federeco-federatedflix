/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/JSF/JSFManagedBean.java to edit this template
 */
package monster.charan.federatedflix.controller;

import jakarta.inject.Named;
import jakarta.enterprise.context.SessionScoped;
import java.io.Serializable;
import org.primefaces.PrimeFaces;

@Named(value = "themeManager")
@SessionScoped
public class ThemeManager implements Serializable {

    private String theme = "arya";
    private String themeColor = "dark";
    private String themeIcon = "moon";

    public String getTheme() {
        return theme;
    }

    public String getThemeColor() {
        return themeColor;
    }

    public String getThemeIcon() {
        return themeIcon;
    }

    public void toggleTheme() {
        if (theme.equals("saga")) {
            theme = "arya";
            PrimeFaces.current().executeScript("PrimeFaces.changeTheme('arya');");
            themeColor = "dark";
            themeIcon = "moon";
        } else {
            theme = "saga";
            PrimeFaces.current().executeScript("PrimeFaces.changeTheme('saga');");
            themeColor = "light";
            themeIcon = "sun";
        }
    }

}
