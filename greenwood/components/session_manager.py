class session_manager:

    def __init__(self, user_id, user_name, user_password_hash, organisation_name, organisation_id, is_admin):
        self.user_id = user_id
        self.user_name = user_name
        self.organisation_id = organisation_id
        self.organisation_name = organisation_name
        self.is_admin = is_admin
        self.save_local_session_data()
        
    # def create_session()

    def save_local_session_data(self):
        pass

    def authenticate_login(self, user_id, user_password_hash):
        # Return true if login correct
        # if x == x: 
        #   return self.y
        # else
        #   return None
        pass

    def logout(self):
        self.session_data = None