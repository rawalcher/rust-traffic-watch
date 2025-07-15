use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration};
use log::{info, error, warn};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::net::SocketAddr;
use crate::network::{send_message, receive_message};
use crate::constants::{CONTROLLER_ADDRESS, JETSON_ADDRESS, PI_ADDRESS};

#[derive(Debug, Clone)]
pub enum ConnectionRole {
    Controller,
    Pi,
    Jetson,
}

impl ConnectionRole {
    pub fn expected_ip(&self) -> &'static str {
        match self {
            ConnectionRole::Controller => CONTROLLER_ADDRESS,
            ConnectionRole::Pi => PI_ADDRESS,
            ConnectionRole::Jetson => JETSON_ADDRESS,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConnectionType {
    Control,
    Data,
}

#[derive(Debug, Clone)]
pub struct PersistentConnection {
    stream: Arc<Mutex<Option<TcpStream>>>,
    address: String,
    pub role: ConnectionRole,
    pub connection_type: ConnectionType,
    max_retries: u32,
    retry_delay: Duration,
}

impl PersistentConnection {
    pub fn new(
        address: String,
        role: ConnectionRole,
        connection_type: ConnectionType,
    ) -> Self {
        Self {
            stream: Arc::new(Mutex::new(None)),
            address,
            role,
            connection_type,
            max_retries: 10,
            retry_delay: Duration::from_secs(2),
        }
    }

    pub fn from_stream(
        stream: TcpStream,
        address: String,
        role: ConnectionRole,
        connection_type: ConnectionType,
    ) -> Self {
        Self {
            stream: Arc::new(Mutex::new(Some(stream))),
            address,
            role,
            connection_type,
            max_retries: 10,
            retry_delay: Duration::from_secs(2),
        }
    }

    pub async fn connect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut attempts = 0;

        while attempts < self.max_retries {
            let connect_future = TcpStream::connect(&self.address);
            let timeout_duration = Duration::from_secs(5);

            match tokio::time::timeout(timeout_duration, connect_future).await {
                Ok(Ok(stream)) => {
                    let mut guard = self.stream.lock().await;
                    *guard = Some(stream);
                    info!("{:?} {:?} connection established to {}", 
                      self.role, self.connection_type, self.address);
                    return Ok(());
                }
                Ok(Err(e)) => {
                    attempts += 1;
                    warn!("{:?} {:?} connection attempt {} failed: {}, retrying...", 
                      self.role, self.connection_type, attempts, e);
                }
                Err(_) => {
                    attempts += 1;
                    warn!("{:?} {:?} connection attempt {} timed out, retrying...", 
                      self.role, self.connection_type, attempts);
                }
            }

            if attempts >= self.max_retries {
                return Err(format!(
                    "Failed to connect {:?} {:?} after {} attempts",
                    self.role, self.connection_type, self.max_retries
                ).into());
            }

            sleep(self.retry_delay).await;
        }

        unreachable!()
    }

    pub async fn send<T: serde::Serialize>(
        &self,
        message: &T,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut guard = self.stream.lock().await;

        if let Some(ref mut stream) = *guard {
            match send_message(stream, message).await {
                Ok(()) => Ok(()),
                Err(e) => {
                    error!("{:?} {:?} send failed: {}", self.role, self.connection_type, e);
                    *guard = None; // Mark connection as broken
                    Err(e)
                }
            }
        } else {
            Err("No active connection".into())
        }
    }

    pub async fn receive<T: for<'de> serde::Deserialize<'de>>(
        &self,
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        let mut guard = self.stream.lock().await;

        if let Some(ref mut stream) = *guard {
            match receive_message(stream).await {
                Ok(message) => Ok(message),
                Err(e) => {
                    error!("{:?} {:?} receive failed: {}", self.role, self.connection_type, e);
                    *guard = None; // Mark connection as broken
                    Err(e)
                }
            }
        } else {
            Err("No active connection".into())
        }
    }

    pub async fn is_connected(&self) -> bool {
        let guard = self.stream.lock().await;
        guard.is_some()
    }

    pub async fn ensure_connected(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_connected().await {
            self.connect().await?;
        }
        Ok(())
    }

    pub async fn disconnect(&self) {
        let mut guard = self.stream.lock().await;
        if let Some(stream) = guard.take() {
            drop(stream);
            info!("{:?} {:?} connection closed", self.role, self.connection_type);
        }
    }
}

pub struct ConnectionManager {
    pub connections: Vec<PersistentConnection>,
}

impl ConnectionManager {
    pub fn new() -> Self {
        Self {
            connections: Vec::new(),
        }
    }

    pub fn add_connection(&mut self, connection: PersistentConnection) {
        self.connections.push(connection);
    }

    pub async fn connect_all(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for connection in &self.connections {
            connection.connect().await?;
        }
        Ok(())
    }

    pub async fn disconnect_all(&self) {
        for connection in &self.connections {
            connection.disconnect().await;
        }
    }

    pub fn get_connection(&self, role: ConnectionRole, conn_type: ConnectionType) -> Option<&PersistentConnection> {
        self.connections.iter().find(|conn| {
            matches!((&conn.role, &conn.connection_type), (r, t) if
                std::mem::discriminant(r) == std::mem::discriminant(&role) &&
                std::mem::discriminant(t) == std::mem::discriminant(&conn_type))
        })
    }
}

pub struct ConnectionListener {
    listener: TcpListener,
    expected_connections: Vec<(ConnectionRole, String)>,
}

impl ConnectionListener {
    pub async fn new(
        bind_address: &str,
        expected_connections: Vec<(ConnectionRole, String)>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(bind_address).await?;
        info!("Listener bound to {}", bind_address);

        Ok(Self {
            listener,
            expected_connections,
        })
    }
    
    pub async fn accept_expected_connections(&self) -> Result<Vec<(ConnectionRole, TcpStream)>, Box<dyn std::error::Error + Send + Sync>> {
        let mut accepted = Vec::new();
        let mut remaining_roles: Vec<ConnectionRole> = self.expected_connections.iter()
            .map(|(role, _)| role.clone())
            .collect();

        while !remaining_roles.is_empty() {
            let (stream, addr) = self.listener.accept().await?;

            if let Some(role) = self.find_matching_role(&addr, &remaining_roles) {
                info!("Accepted {:?} connection from {} (IP validated)", role, addr);
                accepted.push((role.clone(), stream));
                remaining_roles.retain(|r| !matches!((r, &role), (ConnectionRole::Controller, ConnectionRole::Controller) | (ConnectionRole::Pi, ConnectionRole::Pi) | (ConnectionRole::Jetson, ConnectionRole::Jetson)));
            } else {
                warn!("Rejected connection from {} - not from any expected IP", addr);
                // Connection is automatically dropped when stream goes out of scope
            }
        }

        Ok(accepted)
    }

    fn find_matching_role(&self, addr: &SocketAddr, remaining_roles: &[ConnectionRole]) -> Option<ConnectionRole> {
        let client_ip = addr.ip().to_string();

        for role in remaining_roles {
            let expected_ip = role.expected_ip();

            if expected_ip == "127.0.0.1" && (client_ip == "127.0.0.1" || client_ip == "::1") {
                return Some(role.clone());
            }
            
            if client_ip == expected_ip {
                return Some(role.clone());
            }
        }

        None
    }
}