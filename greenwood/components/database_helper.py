"""
Database Helper
________________________________________________________________________
This module has class to handle database operations
"""
from logger import logger
import oracledb
from env_vars import env_dict

class Database(object):

    """
    The `Database` class provides a high-level interface for interacting with the Oracle database.
    """

    def __init__(self):
        """
        Initalizes the database connection
        """
        self.connection = self.__get_connection() #TODO manage failure

    def __get_connection(self):
        """
        Get a connection to the Oracle database using the credentials and connection
        string stored in the environment dictionary.

        :return: A connection object to the Oracle database.
        :rtype: object
        """
        connection = oracledb.connect(
            user=env_dict['DB_USER'],
            password=env_dict['DB_PASSWORD'],
            dsn=env_dict['DB_CONNECTION_STRING'])
        logger.info("Connection to database sucessful")
        return connection

    def __query(self, query):
        """
        Executes a SQL query using the current database connection.

        :param query: The SQL query to execute.
        :type query: str
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query)
        self.connection.commit()
        print("Query Success: ", query)
        
    def create_organization(self, name, address):
        """
        Creates a new organization in the database.

        :param name: The name of the organization.
        :type name: str
        :param address: The address of the organization.
        :type address: str
        """
        self.__query(
            f"""
            INSERT INTO companies (name, address)
            VALUES ('{name}', '{address})')
            """)

    def invite_user(self, email, company_id):
        """
        Invites a user to an organization in the database.

        :param email: The email address of the user to invite.
        :type email: str
        :param company_id: The ID of the organization to invite the user to.
        :type company_id: int
        """
        self.__query(
            f"""
            INSERT INTO invites (email, company_id)
            VALUES ('{email}', {company_id})
            """)
        pass

    def create_user(self, name, email, password):
        """
        Creates a new user account in the database.

        :param name: The name of the user.
        :type name: str
        :param email: The email address of the user.
        :type email: str
        :param password: The password of the user.
        :type password: str
        """
        #TODO hash password MD250
        password_hash = password;
        self.__query(
            f"""
            INSERT INTO accounts (name, email, password_hash)
            VALUES ('{name}', '{email}', '{password_hash}')
            """)
    def add_location(self,company_id,name):
        """
        Adds a new location to an organization in the database.

        :param company_id: The ID of the organization to add the location to.
        :type company_id: int
        :param name: The name of the new location.
        :type name: str
        """
        self.__query(
            f"""
            INSERT INTO locations(company_id, name)
            VALUES ({company_id}, '{name}')
            """)

    def add_activity_record(self, company_id, elephant_frequency, gunshot_frequency, logging_frequency, is_shared, date_start, date_end, time_start, time_end, location_id):
        """
        Adds an activity record for a company to the database.

        :param company_id: The ID of the company.
        :type company_id: int
        :param elephant_frequency: The frequency of elephant activity.
        :type elephant_frequency: float
        :param gunshot_frequency: The frequency of gunshot activity.
        :type gunshot_frequency: float
        :param logging_frequency: The frequency of logging activity.
        :type logging_frequency: float
        :param is_shared: A flag indicating whether the activity record is shared with other companies.
        :type is_shared: bool
        :param date_start: The start date of the activity record.
        :type date_start: str
        :param date_end: The end date of the activity record.
        :type date_end: str
        :param time_start: The start time of the activity record.
        :type time_start: str
        :param time_end: The end time of the activity record.
        :type time_end: str
        :param location_id: The ID of the location.
        :type location_id: int
        """
        self.__query(
            f"""
            INSERT INTO report_summaries (company_id, elephant_frequency, gunshot_frequency, logging_frequency, is_shared, date_start, date_end, time_start, time_end, location_id)
            VALUES ({company_id}, {elephant_frequency}, {gunshot_frequency}, {logging_frequency}, {is_shared}, DATE '{date_start}', DATE'{date_end}',  '{time_start}', '{time_end}',{location_id})
            """)