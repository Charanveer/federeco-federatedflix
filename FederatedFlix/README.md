# FederatedFlix

## Introduction

This is web application allows you to interact with [Federeco](https://github.com/Ach113/federeco) and view movie recommendations based on the trained model.

## Build Requirements

- [JDK 17](https://adoptium.net/temurin/releases/?version=17)

- [NetBeans IDE 17](https://netbeans.apache.org/download/index.html)

- [Sass](https://github.com/sass/sass) Preprocessor for compiling CSS

- [npm](https://github.com/npm/cli) package manager

## Steps

1. Clone the project.

   `git clone https://github.com/charanveer/federatedflix.git`

2. Change dirctory to federatedflex

   ` cd federatedflix`

3. Install dependecies

   `npm install`

4. Open the project in NetBeans, make sure JDK 17 is properly installed and recognized by Netbeans.
5. Click on `Clean and Build` to build the project. This will create a .war file in the `target/` folder.
6. You can deploy the war file on any Jakarta EE application server like Payara Server, Glassfish, Wildfly, TomCat, TomEE etc. This particular project has been tested on Payara.
