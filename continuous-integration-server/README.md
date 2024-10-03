# DD2480 Continuous Integration
This repo contains a basic continuous integration (CI) server made for the course DD2480 at KTH. The server supports compiling and testing Java Maven projects. It can also update the commit status of your repo.


## Setup 
1. Make sure that you have Java, Maven and ngrok installed.
2. Generate a GitHub access token with read/write privlegies for commit statuses for the repo you want to use the server against. Add this token to the string ```GITHUB_TOKEN``` in ContinuousIntegrationServer.java.  
3. Compile and run the program by the commands:
   ```mvn clean install```
   ```mvn exec:java```
4. Start ngrok for port 8080 with:
   ```ngrok http 8080```
5. Paste the URL provided by ngrok into the webhook section of your repo. Make sure to select the JSON format.

Tests can be executed through the ```mvn test``` command.

Javadoc can be generated with mvn javadoc:javadoc.

Further instructions: [Smallest Java CI Repository](https://github.com/KTH-DD2480/smallest-java-ci)


## Details of implementations and unit tests

### Compilation
Github webhooks sends a POST request to our server when someone has made a commit to the remote repository. The server then clones the branch that has been commited to into a temporary folder. The ```compileProject``` function then runs the ```mvn compile``` command in the temporary folder. If no exceptions are thrown and if the exit code of the process is 0, compilation is deemed successful, otherwise not. Testing that the compilation works was done by creating a small mock maven project and running the function on it. We checked that the function returned true, signalling a success, and also that the target folder that maven generates exist after compilation.

### Testing
Triggering Tests: Testing is initiated via GitHub webhooks. Pushing changes to the "assessment" branch in the repository prompts the CI server to automatically execute associated tests. Upon webhook receipt, the CI server extracts repository details, clones, compiles, and runs tests specific to the committed branch. This process ensures changes do not compromise the projects integrity. Testing was done by creating assessment branch and pushed changes for testing, checking CI server log.

### Notifications
When a webhook request has passed through the compilation and testing the step the results are passed to the ```notifyGitHubCommitStatus```. This function uses the GitHub API to set the commit status of the commit that triggered the webhook according to the results of compilation and testing. Authentication is done through pasting your token into the file ```ContinuousIntegrationServer.java```.


## Contributions
Jodie Ooi: 
- Worked on testProject function
- Contributed to readme on testing documentation.

Leo Vainio:
- Wrote the cloneProject function
- Wrote the compileProject function
- Wrote the testProject function
- Set up the repository, organisation and dependencies for the project

Luna Ji Chen:
- Wrote tests for cloneProject and compileProject function
- Worked on setting up the repository for the project

Teodor Morfeldt Gadler: 
- Wrote function and tests for notifications.
- Contributed to readme/essence analysis
- Wrote tests and parts of the functionality for saving build logs

William Carl Vitalis Nordwall:
- Wrote function for maintaining history of builds