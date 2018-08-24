class User:

        def __init__(self, user_is_geo_enabled, user_id, user_name, user_desc, user_screen_name, user_follower_count,
                     user_lang, user_location, user_time_zone, user_friends_count):
            self.user_is_geo_enabled = user_is_geo_enabled;
            self.user_id = user_id
            self.user_name = user_name
            self.user_desc = user_desc
            self.user_screen_name = user_screen_name
            self.user_follower_count = user_follower_count
            self.user_lang = user_lang
            self.user_location = user_location
            self.user_time_zone = user_time_zone
            self.user_friends_count = user_friends_count

        ## Setter Functions:
        def setUserGeoEnabled(self, user_is_geo_enabled):
            self.user_is_geo_enabled = user_is_geo_enabled;

        def setUserName(self, user_name):
            self.user_name = user_name;

        def setUserDesc(self, user_desc):
            self.user_desc = user_desc;

        def setUserScreenName(self, user_screen_name):
            self.user_screen_name = user_screen_name;

        def setUserFollowerCount(self, user_follower_count):
            self.user_follower_count = user_follower_count;

        def setUserLang(self, user_lang):
            self.user_lang = user_lang;

        def setUserLocation(self, user_location):
            self.user_location = user_location;

        def setUserTimeZone(self, user_time_zone):
            self.user_time_zone = user_time_zone;

        def setUserFriendsCount(self, user_friends_count):
            self.user_friends_count = user_friends_count;
