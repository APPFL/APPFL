from appfl.login_manager import LoginManager

def auth():
    while True:
        mode = input(
            "Please select whether you are authenticating for the federated learning server or client (Enter 1 or 2)\n"
            "-------------------------------------------------------------------------------------------------------\n"
            "1. Federated learning server\n"
            "2. Federated learning client\n"
            "-------------------------------------------------------------------------------------------------------\n"
        )
        if mode in ("1", "2"):
            break
        else:
            print(
                "Invalid input, please try again\n"
                "-------------------------------------------------------------------------------------------------------\n"
            )
    is_fl_server = mode == "1"
    login_manager = LoginManager(is_fl_server=is_fl_server)
    if login_manager.ensure_logged_in():
        action = input(
            f"You have already logged in as a federated learning {'server' if is_fl_server else 'client'}\n"
            "You can either logout (1), change to another account (2), or just exit (3)\n"
            "1. Logout\n"
            "2. Change to another account\n"
            "3. Exit\n"
        )
        if not action in ["1", "2", "3"]:
            print("Invalid input, exiting...")
            return
        if action == "1":
            login_manager.logout()
        elif action == "2":
            login_manager.logout()
            login_manager = LoginManager(is_fl_server=is_fl_server)
    else:
        print(
            f"Successfully logged in as a federated learning {'server' if is_fl_server else 'client'}\n"
        )
