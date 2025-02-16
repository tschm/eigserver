import pyarrow as pa
import pyarrow.flight as fl
import polars as pl
import numpy as np
from scipy.linalg import eigvals


class EigenvalueServer(fl.FlightServerBase):
    def __init__(self, host, port):
        # Bind using a valid URI format
        uri = f"grpc+tcp://{host}:{port}"
        super().__init__(uri)

    def do_get(self, context, ticket):
        # The ticket contains the matrix data as Arrow data
        table = pa.ipc.open_stream(ticket).read_all()

        # Convert Arrow Table to Polars DataFrame
        df = pl.from_arrow(table)

        # Process each matrix/vector (assuming 'data' column contains flattened arrays)
        result = []

        for row in df.iter_rows():
            matrix_data = row[0]  # Flattened data of matrix/vector
            shape = row[1]  # Shape info (number of rows and columns or length)

            # Convert the flattened data into a NumPy array based on shape
            matrix = np.array(matrix_data).reshape(shape)

            # Compute eigenvalue (or any other computation)
            eigenvalues = eigvals(matrix)

            # Extract the smallest eigenvalue
            smallest_eigenvalue = np.min(np.real(eigenvalues))

            result.append(smallest_eigenvalue)

        # Prepare Arrow Table with the results
        result_array = pa.array(result, type=pa.float64())
        result_table = pa.table({"smallest_eigenvalue": result_array})

        return fl.RecordBatchStream(result_table)

    def do_put(self, context, descriptor, reader):
        # Read the matrix data from the client
        # batch = reader.read_all()
        return fl.FlightDescriptor.for_command("Compute smallest eigenvalue")


# Start the server
def start_server():
    server = EigenvalueServer("0.0.0.0", 5005)  # Bind to localhost and port 5005
    print("Starting Eigenvalue Flight server on port 5005...")
    server.serve()


if __name__ == "__main__":
    start_server()
