import requests
import time

class FoldseekClient:
    def __init__(self):
        self.base_url = "https://search.foldseek.com/api"
        self.tickets = list()
        self.ticket = None

    def submit_job(self, file_path, databases, mode='all'):
        try:
            files = {'q': open(file_path, 'rb')}
            data = {
                'database[]': databases,
                'mode': mode,
            }
            response = requests.post(f"{self.base_url}/ticket", files=files, data=data)

            # data = {
            #     'q': file_path,
            #     'database[]': databases,
            #     'mode': mode,
            # }
            # response = requests.post(f"{self.base_url}/ticket", data=data)

            if response.status_code == 200:
                ticket_info = response.json()
                self.ticket = ticket_info

                print("Job submitted successfully. Ticket:", self.ticket)
            else:
                print("Error submitting job. Status code:", response.status_code)
        except Exception as e:
            print("Error:", e)

    def check_status(self):
        try:
            if self.ticket:
                response = requests.get(f"{self.base_url}/ticket/{self.ticket['id']}")
                if response.status_code == 200:
                    status_info = response.json()
                    print("Job status:", status_info['status'])
                    if status_info['status'] == "ERROR":
                        print("job status: ERROR")
                    return status_info['status']
                else:
                    print("Error checking job status. Status code:", response.status_code)
            else:
                print("No ticket found. Please submit a job first.")
        except Exception as e:
            print("Error:", e)

    def fetch_result(self):
        try:
            if self.ticket:
                response = requests.get(f"{self.base_url}/result/{self.ticket}")
                if response.status_code == 200:
                    result_info = response.json()
                    print("Job result:", result_info['result'])
                else:
                    print("Error fetching job result. Status code:", response.status_code)
            else:
                print("No ticket found. Please submit a job first.")
        except Exception as e:
            print("Error:", e)

    def get_tickets_status(self, ticket_ids):
        try:
            data = {'tickets[]': ticket_ids}
            response = requests.post(f"{self.base_url}/tickets", data=data)
            if response.status_code == 200:
                tickets_info = response.json()
                for ticket_info in tickets_info:
                    print("Ticket ID:", ticket_info['id'], "Status:", ticket_info['status'])
            else:
                print("Error getting tickets status. Status code:", response.status_code)
        except Exception as e:
            print("Error:", e)

    def get_alignment_results(self, ticket_id, entry_number):
        try:
            response = requests.get(f"{self.base_url}/result/{ticket_id}/{entry_number}")
            if response.status_code == 200:
                alignment_info = response.json()
                print("Query Header:", alignment_info['query']['header'])
                print("Query Sequence:", alignment_info['query']['sequence'])
                for result in alignment_info['results']:
                    print("Database:", result['db'])
                    for alignment in result['alignments']:
                        print("Alignment:")
                        print("  Query:", alignment['query'])
                        print("  Target:", alignment['target'])
                        print("  Sequence Identity:", alignment['seqId'])
                        print("  Alignment Length:", alignment['alnLength'])
                        print("  Missmatches:", alignment['missmatches'])
                        print("  Gaps Opened:", alignment['gapsopened'])
                        print("  Query Start Position:", alignment['qStartPos'])
                        print("  Query End Position:", alignment['qEndPos'])
                        print("  Database Start Position:", alignment['dbStartPos'])
                        print("  Database End Position:", alignment['dbEndPos'])
                        print("  E-value:", alignment['eval'])
                        print("  Score:", alignment['score'])
                        print("  Query Length:", alignment['qLen'])
                        print("  Database Length:", alignment['dbLen'])
                        print()
            else:
                print("Error getting alignment results. Status code:", response.status_code)
        except Exception as e:
            print("Error:", e)

    def download_alignment_results(self, ticket_id):
        try:
            response = requests.get(f"{self.base_url}/result/download/{ticket_id}", stream=True)
            if response.status_code == 200:
                with open(f"{ticket_id}_results.blasttab", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print("Alignment results downloaded successfully.")
            elif response.status_code == 400:
                print("Invalid ticket. Please check the ticket ID.")
            else:
                print("Error downloading alignment results. Status code:", response.status_code)
        except Exception as e:
            print("Error:", e)

    def submit_job_and_get_results(self, file_path, databases, mode='all', check_interval=1):
        try:
            # Submit job
            self.submit_job(file_path, databases, mode)
            if self.ticket:
                # Poll status until it becomes "COMPLETE"
                while True:
                    status = self.check_status()
                    if status == "COMPLETE":
                        # Download results
                        self.get_alignment_results(self.ticket['id'], 0)
                        break
                    elif status == "ERROR":
                        print("Job encountered an error. Please check the job status.")
                        break
                    else:
                        time.sleep(check_interval)
            else:
                print("No ticket found. Please submit a job first.")
        except Exception as e:
            print("Error:", e)

# Example usage:
if __name__ == "__main__":
    client = FoldseekClient()
    file = '' # Replace with your FASTA file path
    databases = ['pdb100', 'cath40']
    client.submit_job_and_get_results(file, databases, mode="summary")



